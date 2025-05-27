# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp
import newton
import newton.core.articulation

from ik_objectives import IKObjective, JacobianMode

###########################################################################
# Modular IK Solver
###########################################################################

# Default tile constants - will be overridden by specialized classes
TILE_THREADS = wp.constant(32)

@wp.kernel
def update_joint_positions(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array2d(dtype=wp.float32),
    step_size: float,
    coords_per_env: int,
):
    global_joint_idx = wp.tid()
    
    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env
    
    joint_q[global_joint_idx] -= step_size * delta_q[env_idx, joint_idx_in_env]

class ModularLevenbergMarquardtIKSolver:
    """Base class for modular IK solver - will be specialized with correct tile dimensions"""
    
    # These will be overridden in specialized classes
    TILE_COORDS = None
    TILE_RESIDUALS = None
    TILE_THREADS = TILE_THREADS
    
    def __init__(self, model, num_envs, objectives, damping=1e-4, jacobian_mode=JacobianMode.AUTODIFF):
        self.model = model
        self.num_envs = num_envs
        self.objectives = objectives
        self.damping = damping
        self.jacobian_mode = jacobian_mode

        
        # Calculate dimensions
        self.coords = model.joint_coord_count
        self.coords_per_env = self.coords // num_envs
        
        # Calculate residual offsets for each objective
        self.residual_offsets = []
        current_offset = 0
        for obj in objectives:
            self.residual_offsets.append(current_offset)
            current_offset += obj.residual_dim()
        self.num_residuals_per_env = current_offset
        
        # Verify dimensions match tile constants
        if self.TILE_COORDS is not None:
            assert self.coords_per_env == self.TILE_COORDS
        if self.TILE_RESIDUALS is not None:
            assert self.num_residuals_per_env == self.TILE_RESIDUALS
        
        # Pre-allocate arrays
        self.state = self.model.state()
        self.residuals = wp.zeros((self.num_envs, self.num_residuals_per_env), dtype=wp.float32, requires_grad=True)
        
        damping_diag_np = np.full(self.coords_per_env, self.damping, dtype=np.float32)
        self.damping_diag_wp = wp.array(damping_diag_np, dtype=wp.float32)
        
        self.jacobian = wp.zeros((self.num_envs, self.num_residuals_per_env, self.coords_per_env), dtype=wp.float32)
        self.tape = wp.Tape()
        
        # Arrays for solve
        self.delta_q_per_env = wp.zeros((self.num_envs, self.coords_per_env), dtype=wp.float32)
        self.residuals_3d = wp.zeros((self.num_envs, self.num_residuals_per_env, 1), dtype=wp.float32)
        
        # CUDA graph support
        self.use_cuda_graph = wp.get_device().is_cuda
        self.graph = None
    
    def compute_residuals(self):
        """Compute residuals from all objectives"""
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        self.residuals.zero_()
        
        for obj, offset in zip(self.objectives, self.residual_offsets):
            obj.compute_residuals(self.state, self.model, self.residuals, offset)
        
        return self.residuals

    def compute_jacobian(self):
        """Compute Jacobian from all objectives"""
        self.jacobian.zero_()
        
        if self.jacobian_mode == JacobianMode.AUTODIFF:
            # Original autodiff path
            with self.tape:
                residuals_2d = self.compute_residuals()
                current_residuals_wp = residuals_2d.flatten()
            
            self.tape.outputs = [current_residuals_wp]
            
            for obj, offset in zip(self.objectives, self.residual_offsets):
                obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset)
                self.tape.zero()
        
        elif self.jacobian_mode == JacobianMode.ANALYTIC:
            # Analytic path
            for obj, offset in zip(self.objectives, self.residual_offsets):
                if obj.supports_analytic():
                    obj.compute_jacobian_analytic(self.state, self.model, self.jacobian, offset)
                else:
                    raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")
        
        elif self.jacobian_mode == JacobianMode.MIXED:
            # Mixed mode - use analytic where available
            # First, need autodiff tape for objectives that don't support analytic
            need_autodiff = any(not obj.supports_analytic() for obj in self.objectives)
            
            if need_autodiff:
                with self.tape:
                    residuals_2d = self.compute_residuals()
                    current_residuals_wp = residuals_2d.flatten()
                self.tape.outputs = [current_residuals_wp]
            
            for obj, offset in zip(self.objectives, self.residual_offsets):
                if obj.supports_analytic():
                    obj.compute_jacobian_analytic(self.state, self.model, self.jacobian, offset)
                else:
                    obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset)
        
        return self.jacobian

    def _solve_iteration(self, step_size=1.0):
        """Single iteration of the solver"""
        residuals_per_env = self.compute_residuals()
        jacobian_per_env = self.compute_jacobian()
        
        # Reshape residuals for tile operations
        residuals_flat = residuals_per_env.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)
        
        self.delta_q_per_env.zero_()
        
        # For the base class, we can't use tiles (no constants)
        # The specialized class will override this
        if self.TILE_COORDS is not None and self.TILE_RESIDUALS is not None:
            # Use specialized kernel that will be generated
            self._solve_with_tiles(jacobian_per_env, self.residuals_3d, 
                                 self.damping_diag_wp, self.delta_q_per_env)
        
        # Update joint positions
        wp.launch(
            update_joint_positions,
            dim=self.model.joint_coord_count,
            inputs=[
                self.model.joint_q,
                self.delta_q_per_env,
                step_size,
                self.coords_per_env
            ]
        )
    
    def _solve_with_tiles(self, jacobian, residuals, damping, delta_q):
        """To be implemented by specialized class with proper tile constants"""
        raise NotImplementedError("This method should be overridden by specialized solver")
    
    def solve(self, iterations=10, step_size=1.0):
        """Solve the IK problem"""
        # Capture graph on first use
        if self.use_cuda_graph and self.graph is None:
            with wp.ScopedCapture() as capture:
                self._solve_iteration(step_size=step_size)
            self.graph = capture.graph
        
        for i in range(iterations):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._solve_iteration(step_size=step_size)


def create_ik_solver(model, num_envs, objectives, damping=1e-4, jacobian_mode=JacobianMode.AUTODIFF):
    """
    Factory function to create a specialized IK solver with correct tile dimensions
    
    Args:
        model: The articulation model
        num_envs: Number of parallel environments
        objectives: List of IKObjective instances
        damping: Damping parameter for Levenberg-Marquardt
        jacobian_mode: How to compute Jacobians (AUTODIFF, ANALYTIC, or MIXED)
    
    Returns:
        Specialized ModularLevenbergMarquardtIKSolver instance
    """
    # Calculate dimensions
    coords_per_env = model.joint_coord_count // num_envs
    total_residuals = sum(obj.residual_dim() for obj in objectives)
    
    # Create specialized kernel for this configuration
    @wp.kernel
    def solve_normal_equations_tiled_specialized(
        jacobians: wp.array3d(dtype=wp.float32),
        residuals: wp.array3d(dtype=wp.float32),
        damping_diag: wp.array1d(dtype=wp.float32),
        delta_q: wp.array2d(dtype=wp.float32),
    ):
        env_idx = wp.tid()
        
        # Use the constants defined in the class
        J = wp.tile_load(jacobians[env_idx], shape=(SpecializedSolver.TILE_RESIDUALS, SpecializedSolver.TILE_COORDS))
        r = wp.tile_load(residuals[env_idx], shape=(SpecializedSolver.TILE_RESIDUALS, 1))
        
        Jt = wp.tile_transpose(J)
        
        JtJ = wp.tile_zeros(shape=(SpecializedSolver.TILE_COORDS, SpecializedSolver.TILE_COORDS), dtype=wp.float32)
        wp.tile_matmul(Jt, J, JtJ)
        
        damping_tile = wp.tile_load(damping_diag, shape=(SpecializedSolver.TILE_COORDS,))
        A = wp.tile_diag_add(JtJ, damping_tile)
        
        Jtr = wp.tile_zeros(shape=(SpecializedSolver.TILE_COORDS, 1), dtype=wp.float32)
        wp.tile_matmul(Jt, r, Jtr)
        
        Jtr_1d = wp.tile_zeros(shape=(SpecializedSolver.TILE_COORDS,), dtype=wp.float32)
        for i in range(SpecializedSolver.TILE_COORDS):
            Jtr_1d[i] = Jtr[i, 0]
        
        L = wp.tile_cholesky(A)
        
        solution = wp.tile_zeros(shape=(SpecializedSolver.TILE_COORDS,), dtype=wp.float32)
        wp.tile_assign(solution, Jtr_1d, offset=(0,))
        wp.tile_cholesky_solve(L, solution)
        
        wp.tile_store(delta_q[env_idx], solution)
    
    # Create specialized solver class
    class SpecializedSolver(ModularLevenbergMarquardtIKSolver):
        TILE_COORDS = wp.constant(coords_per_env)
        TILE_RESIDUALS = wp.constant(total_residuals)
        TILE_THREADS = wp.constant(32)
        
        def _solve_with_tiles(self, jacobian, residuals, damping, delta_q):
            wp.launch_tiled(
                solve_normal_equations_tiled_specialized,
                dim=[self.num_envs],
                inputs=[jacobian, residuals, damping, delta_q],
                block_dim=self.TILE_THREADS
            )
    
    return SpecializedSolver(model, num_envs, objectives, damping, jacobian_mode)