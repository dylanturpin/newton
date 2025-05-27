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


###########################################################################
# Example Jacobian with Joint Limits
#
# Demonstrates how to compute the Jacobian of a multi-valued function
# with joint limit constraints. Uses a unified residual approach where
# joint limit violations are treated as additional residuals.
#
###########################################################################

import math

import numpy as np

import warp as wp

import newton
import newton.core.articulation
import newton.examples
import newton.utils


@wp.kernel
def compute_ee_position_residuals(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.array(dtype=wp.vec3),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    residuals: wp.array(dtype=wp.float32),
    offset: int,
):
    tid = wp.tid()
    
    # Compute end-effector position
    ee_pos = wp.transform_point(body_q[tid * num_links + ee_link_index], ee_link_offset)
    
    # Compute position error and write to residuals
    error = target_pos[tid] - ee_pos
    residuals[offset + tid * 3 + 0] = error[0]
    residuals[offset + tid * 3 + 1] = error[1]
    residuals[offset + tid * 3 + 2] = error[2]


@wp.kernel
def compute_joint_limit_residuals(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    dof: int,
    residuals: wp.array(dtype=wp.float32),
    offset: int,
    weight: float,
):
    tid = wp.tid()
    
    q = joint_q[tid]
    lower = joint_limit_lower[tid % dof]
    upper = joint_limit_upper[tid % dof]
    
    # Compute violation: max(0, q-upper) + max(0, lower-q)
    upper_violation = wp.max(0.0, q - upper)
    lower_violation = wp.max(0.0, lower - q)
    
    residuals[offset + tid] = weight * (upper_violation + lower_violation)


class LevenbergMarquardtIKSolver:
    def __init__(self, model, num_envs, ee_link_index, ee_link_offset, num_links, damping=1e-4, joint_limit_weight=0.1):
        self.model = model
        self.num_envs = num_envs
        self.ee_link_index = ee_link_index
        self.ee_link_offset = ee_link_offset
        self.num_links = num_links
        self.damping = damping
        self.joint_limit_weight = joint_limit_weight
        self.dof = len(model.joint_q) // num_envs
        
        # Total number of residuals: 3 per env (EE position) + 1 per joint (limit violations)
        self.total_joints = len(model.joint_q)
        self.num_residuals = 3 * num_envs + self.total_joints
        
        # Create internal state for FK evaluation
        self.state = self.model.state()
        
        # Unified residuals array
        self.residuals = wp.zeros(self.num_residuals, dtype=wp.float32, requires_grad=True)
        
        # Offsets for different residual groups
        self.ee_residual_offset = 0
        self.joint_limit_residual_offset = 3 * num_envs
        
    def compute_residuals(self, target_pos):
        # Evaluate forward kinematics
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        # Compute EE position residuals
        wp.launch(
            compute_ee_position_residuals,
            dim=self.num_envs,
            inputs=[
                self.state.body_q,
                target_pos,
                self.num_links,
                self.ee_link_index,
                self.ee_link_offset,
                self.residuals,
                self.ee_residual_offset,
            ],
        )
        
        # Compute joint limit residuals
        wp.launch(
            compute_joint_limit_residuals,
            dim=self.total_joints,
            inputs=[
                self.model.joint_q,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.dof,
                self.residuals,
                self.joint_limit_residual_offset,
                self.joint_limit_weight,
            ],
        )
        
        return self.residuals
        
    def compute_jacobian(self, target_pos):
        """
        Compute full Jacobian matrix using parallel environment optimization.
        
        This method reduces backward passes from num_residuals to just 4:
        - 3 passes for EE position residuals (x, y, z components across all environments)
        - 1 pass for ALL joint limit residuals (exploiting diagonal structure)
        
        Key optimizations:
        1. EE residuals: Compute same component (x/y/z) across all environments simultaneously
        2. Block-diagonal structure: Each environment's residual only depends on its own joints
        3. Joint limits: All computed in one pass due to diagonal Jacobian structure
        """
        # Compute full Jacobian matrix for all residuals
        jacobian = np.empty((self.num_residuals, self.total_joints), dtype=np.float32)
        
        tape = wp.Tape()
        with tape:
            self.compute_residuals(target_pos)
        
        # 1. Compute EE position residuals in parallel (3 passes for x, y, z components)
        for ee_component in range(3):  # x, y, z
            e = np.zeros(self.num_residuals, dtype=np.float32)
            # Activate the same component (x, y, or z) for all environments
            for env_idx in range(self.num_envs):
                global_residual_idx = self.ee_residual_offset + env_idx * 3 + ee_component
                e[global_residual_idx] = 1.0
            
            e_wp = wp.array(e, dtype=wp.float32)
            tape.backward(grads={self.residuals: e_wp})
            q_grad = tape.gradients[self.model.joint_q].numpy()
            
            # The gradient is block-structured: reshape and assign to correct environments
            q_grad_reshaped = q_grad.reshape(self.num_envs, self.dof)
            
            for env_idx in range(self.num_envs):
                global_residual_idx = self.ee_residual_offset + env_idx * 3 + ee_component
                # Each environment's residual only depends on that environment's joints
                jacobian[global_residual_idx, :] = 0.0  # Zero out first
                start_joint = env_idx * self.dof
                end_joint = (env_idx + 1) * self.dof
                jacobian[global_residual_idx, start_joint:end_joint] = q_grad_reshaped[env_idx, :]
            
            tape.zero()
        
        # 2. Compute ALL joint limit residuals in one pass (diagonal structure)
        e = np.zeros(self.num_residuals, dtype=np.float32)
        # Activate all joint limit residuals at once
        e[self.joint_limit_residual_offset:] = 1.0
        
        e_wp = wp.array(e, dtype=wp.float32)
        tape.backward(grads={self.residuals: e_wp})
        q_grad = tape.gradients[self.model.joint_q].numpy()
        
        # Fill the diagonal structure - each joint limit residual only affects its own joint
        for joint_idx in range(self.total_joints):
            residual_idx = self.joint_limit_residual_offset + joint_idx
            jacobian[residual_idx, :] = 0.0  # Zero out the row first
            jacobian[residual_idx, joint_idx] = q_grad[joint_idx]  # Only diagonal entry
        
        return jacobian
        
    def solve(self, target_pos, max_iters=10, tolerance=1e-4, step_size=1.0):
        """Solve IK using Levenberg-Marquardt with joint limits"""
        target_pos_wp = wp.array(target_pos, dtype=wp.vec3)
        
        for i in range(max_iters):
            # Compute residuals and Jacobian
            with wp.ScopedTimer("compute_residuals"):
                residuals = self.compute_residuals(target_pos_wp).numpy()
            with wp.ScopedTimer("compute_jacobians"):
                jacobian = self.compute_jacobian(target_pos_wp)
            
            # Extract EE position errors for convergence check
            ee_errors = residuals[:3*self.num_envs].reshape(self.num_envs, 3)
            error_norm = np.linalg.norm(ee_errors)
            
            if error_norm < tolerance:
                print(f"  iter {i}: converged, error={error_norm:.6f}")
                return True, error_norm
            else:
                # Check joint limit violations
                joint_violations = residuals[3*self.num_envs:]
                max_violation = np.max(np.abs(joint_violations)) / self.joint_limit_weight
                print(f"  iter {i}: error={error_norm:.6f}, max joint violation={max_violation:.6f}")
                
            # Levenberg-Marquardt update: (J^T J + λI) δq = J^T r
            JtJ = jacobian.T @ jacobian
            Jtr = jacobian.T @ residuals
            
            # Add damping
            A = JtJ + self.damping * np.eye(self.total_joints)
            
            # Solve for joint angle updates
            delta_q = np.linalg.solve(A, Jtr)
            
            # Update joint angles
            self.model.joint_q = wp.array(
                self.model.joint_q.numpy() - step_size * delta_q,
                dtype=wp.float32,
                requires_grad=True,
            )
            
        print(f"  Did not converge after {max_iters} iterations")
        return False, error_norm


class Example:
    def __init__(self, stage_path="example_jacobian_ik.usd", num_envs=10):
        rng = np.random.default_rng(42)

        self.num_envs = num_envs

        fps = 60
        self.frame_dt = 1.0 / fps

        self.render_time = 0.0

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_urdf(
            newton.examples.get_asset("cartpole.urdf"),
            articulation_builder,
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        builder = newton.ModelBuilder()

        self.num_links = len(articulation_builder.joint_type)
        # use the last link as the end-effector
        self.ee_link_index = self.num_links - 1
        self.ee_link_offset = wp.vec3(0.0, 0.0, 1.0)

        self.dof = len(articulation_builder.joint_q)

        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 0.0, 2.0))

        self.target_origin = []
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(positions[i], wp.quat_identity()))
            self.target_origin.append((positions[i][0]+0.12, positions[i][1], positions[i][2]))
            # joint initial positions
            builder.joint_q[-3:] = rng.uniform(-0.5, 0.5, size=3)
        self.target_origin = np.array(self.target_origin)

        # finalize model
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = False

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=2.0)
        else:
            self.renderer = None

        # Create IK solver with joint limits
        self.ik_solver = LevenbergMarquardtIKSolver(
            self.model, 
            self.num_envs, 
            self.ee_link_index, 
            self.ee_link_offset, 
            self.num_links,
            joint_limit_weight=0.1
        )
        
        # Initialize target positions
        self.target_pos = self.target_origin.copy()

        self.profiler = {}

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.render_time)
            # Get current state from solver
            newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.ik_solver.state)
            self.renderer.render(self.ik_solver.state)
            self.renderer.render_points("targets", self.target_pos, radius=0.05)
            
            # Extract EE positions for visualization
            ee_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
            for i in range(self.num_envs):
                body_tf = self.ik_solver.state.body_q.numpy()[i * self.num_links + self.ee_link_index]
                ee_pos[i] = wp.transform_point(
                    wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                    self.ee_link_offset
                )
            
            self.renderer.render_points("ee_pos", ee_pos, radius=0.05)
            self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_jacobian_ik.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_targets", type=int, default=100, help="Total number of different targets to try.")
    parser.add_argument("--num_envs", type=int, default=20, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    rng = np.random.default_rng(42)

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for target_idx in range(args.num_targets):
            # Set new random target for all environments (keep x coordinate fixed)
            example.target_pos = example.target_origin.copy()
            example.target_pos[:, 1:] += rng.uniform(-0.5, 0.5, size=(example.num_envs, 2))
            
            # Solve IK once for this target set (iterations happen inside solver)
            with wp.ScopedTimer("solve"):
                converged, final_error = example.ik_solver.solve(example.target_pos)
            
            example.render()
            print(f"Target {target_idx}: converged={converged}, final_error={final_error:.6f}")

        if example.renderer:
            example.renderer.save()
