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

import math
import numpy as np
import warp as wp
import newton
import newton.core.articulation
import pyglet.window.mouse

###########################################################################
# Geometry Utilities
###########################################################################

def rotate_vector_by_quaternion(v, q):
    q_vec = q[:3]
    q_scalar = q[3]
    v_prime = 2.0 * np.dot(q_vec, v) * q_vec \
          + (q_scalar**2 - np.dot(q_vec, q_vec)) * v \
          + 2.0 * q_scalar * np.cross(q_vec, v)
    return v_prime


def get_rotation_quaternion(v_from, v_to):
    """Computes quaternion to rotate vector v_from to v_to."""
    v_from = np.asarray(v_from, dtype=np.float32)
    v_to = np.asarray(v_to, dtype=np.float32)

    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)

    if norm_from < 1e-6 or norm_to < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    v_from_norm = v_from / norm_from
    v_to_norm = v_to / norm_to

    dot_product = np.dot(v_from_norm, v_to_norm)

    if dot_product > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    elif dot_product < -0.999999:
        axis = np.cross(v_from_norm, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from_norm, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        angle = math.pi
    else:
        axis = np.cross(v_from_norm, v_to_norm)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(dot_product)

    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def closest_point_on_line_to_ray(line_origin, line_dir, ray_origin, ray_dir):
    """Finds the parameter t1 for the point on line that is closest to ray."""
    line_dir = line_dir / np.linalg.norm(line_dir)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    r = line_origin - ray_origin
    d1d1 = 1.0 
    d2d2 = 1.0 
    d1d2 = np.dot(line_dir, ray_dir)
    
    det = d1d2 * d1d2 - d1d1 * d2d2 

    if abs(det) < 1e-6:
        t1 = np.dot(ray_origin - line_origin, line_dir) / d1d1
        return t1

    t1 = (np.dot(r, line_dir) - np.dot(r, ray_dir) * d1d2) / det
    return t1


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    L = sphere_center - ray_origin
    tca = np.dot(L, ray_dir)
    d2 = np.dot(L, L) - tca * tca
    if d2 > sphere_radius**2: return float('inf')
    thc = np.sqrt(sphere_radius**2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    if t0 < 0 and t1 < 0: return float('inf')
    return t1 if t0 < 0 else t0


def ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, radius):
    ca = p_end - p_start
    oa = ray_origin - p_start
    caca = np.dot(ca, ca)
    card = np.dot(ca, ray_dir)
    caoa = np.dot(ca, oa)
    a = caca - card * card
    b = caca * np.dot(oa, ray_dir) - caoa * card
    c = caca * np.dot(oa, oa) - caoa * caoa - radius * radius * caca
    h = b * b - a * c
    if h < 0.0: return float('inf')
    t = (-b - np.sqrt(h)) / a
    y = caoa + t * card
    if y > 0.0 and y < caca: return t if t > 0 else float('inf')
    t_cap1 = ray_sphere_intersect(ray_origin, ray_dir, p_start, radius)
    t_cap2 = ray_sphere_intersect(ray_origin, ray_dir, p_end, radius)
    if y <= 0.0: return t_cap1 if t_cap1 > 0 else float('inf')
    elif y >= caca: return t_cap2 if t_cap2 > 0 else float('inf')
    return float('inf')


###########################################################################
# IK Solver
###########################################################################

# Tile constants for IK solver - increased for multi-target support
TILE_DOF = wp.constant(52)  # DOF per environment
TILE_RESIDUALS = wp.constant(70)  # 18 EE position + 52 joint limits
TILE_THREADS = wp.constant(32)

@wp.kernel
def compute_ee_error_norm_squared(
    residuals: wp.array2d(dtype=wp.float32),  # (num_envs, num_residuals_per_env)
    error_norm_squared: wp.array(dtype=wp.float32),  # (1,) - output
    num_ees: int,
):
    # Use thread index to process different environments in parallel
    env_idx = wp.tid()
    
    # Compute squared norm of EE position errors (first 3*num_ees residuals)
    local_sum = wp.float32(0.0)
    for i in range(3 * num_ees):
        r = residuals[env_idx, i]
        local_sum += r * r
    
    # Atomically add to global sum
    wp.atomic_add(error_norm_squared, 0, local_sum)


@wp.kernel
def update_joint_positions(
    joint_q: wp.array(dtype=wp.float32),        # Current joint positions
    delta_q: wp.array2d(dtype=wp.float32),      # (num_envs, dof) - delta from solver
    step_size: float,
    dof_per_env: int,
):
    # Global joint index
    global_joint_idx = wp.tid()
    
    # Calculate which environment and which joint within that environment
    env_idx = global_joint_idx / dof_per_env
    joint_idx_in_env = global_joint_idx % dof_per_env
    
    # Update the joint position
    joint_q[global_joint_idx] -= step_size * delta_q[env_idx, joint_idx_in_env]

@wp.kernel
def solve_normal_equations_tiled(
    jacobians: wp.array3d(dtype=wp.float32),     # (num_envs, num_residuals, dof)
    residuals: wp.array3d(dtype=wp.float32),     # (num_envs, num_residuals, 1)
    damping_diag: wp.array1d(dtype=wp.float32),  # (dof,) - pre-computed
    delta_q: wp.array2d(dtype=wp.float32),       # (num_envs, dof) - output
):
    env_idx = wp.tid()
    
    # Load Jacobian for this environment as a tile
    J = wp.tile_load(jacobians[env_idx], shape=(TILE_RESIDUALS, TILE_DOF))
    
    # Load residuals for this environment as a 2D tile  
    r = wp.tile_load(residuals[env_idx], shape=(TILE_RESIDUALS, 1))
    
    # Compute J^T (transpose)
    Jt = wp.tile_transpose(J)
    
    # Compute J^T J
    JtJ = wp.tile_zeros(shape=(TILE_DOF, TILE_DOF), dtype=wp.float32)
    wp.tile_matmul(Jt, J, JtJ)
    
    # Load pre-computed damping diagonal and add to JtJ
    damping_tile = wp.tile_load(damping_diag, shape=(TILE_DOF,))
    A = wp.tile_diag_add(JtJ, damping_tile)
    
    # Compute J^T r using tile_matmul
    Jtr = wp.tile_zeros(shape=(TILE_DOF, 1), dtype=wp.float32)
    wp.tile_matmul(Jt, r, Jtr)
    
    # Extract column 0 from Jtr to get 1D tile for cholesky_solve
    Jtr_1d = wp.tile_zeros(shape=(TILE_DOF,), dtype=wp.float32)
    for i in range(TILE_DOF):
        Jtr_1d[i] = Jtr[i, 0]
    
    # Solve A * delta_q = Jtr using Cholesky decomposition
    L = wp.tile_cholesky(A)
    
    # Solve for delta_q (tile_cholesky_solve modifies the second argument in-place)
    solution = wp.tile_zeros(shape=(TILE_DOF,), dtype=wp.float32)
    wp.tile_assign(solution, Jtr_1d, offset=(0,))
    wp.tile_cholesky_solve(L, solution)
    
    # Store result
    wp.tile_store(delta_q[env_idx], solution)

@wp.kernel
def compute_ee_position_residuals(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.array(dtype=wp.vec3),  # Flattened: (num_envs * num_ees,)
    num_links: int,
    ee_link_indices: wp.array(dtype=int),  # (num_ees,)
    ee_link_offsets: wp.array(dtype=wp.vec3),  # (num_ees,)
    residuals: wp.array2d(dtype=wp.float32),  # (num_envs, num_residuals_per_env)
    num_ees: int,
):
    tid = wp.tid()
    env_idx = tid / num_ees
    ee_idx = tid % num_ees
    
    link_idx = ee_link_indices[ee_idx]
    ee_pos = wp.transform_point(body_q[env_idx * num_links + link_idx], ee_link_offsets[ee_idx])
    error = target_pos[tid] - ee_pos
    
    # Write to the appropriate residual indices
    base_idx = ee_idx * 3
    residuals[env_idx, base_idx + 0] = error[0]
    residuals[env_idx, base_idx + 1] = error[1]
    residuals[env_idx, base_idx + 2] = error[2]


@wp.kernel
def compute_joint_limit_residuals(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    dof_per_env: int,
    residuals: wp.array2d(dtype=wp.float32),  # (num_envs, num_residuals_per_env)
    weight: float,
    num_ees: int,
):
    global_joint_idx = wp.tid()
    
    env_idx = global_joint_idx / dof_per_env
    joint_idx_in_env = global_joint_idx % dof_per_env
    
    q = joint_q[global_joint_idx]
    lower = joint_limit_lower[joint_idx_in_env]
    upper = joint_limit_upper[joint_idx_in_env]
    
    upper_violation = wp.max(0.0, q - upper)
    lower_violation = wp.max(0.0, lower - q)
    
    # Joint limit residuals start after EE residuals
    residuals[env_idx, 3 * num_ees + joint_idx_in_env] = weight * (upper_violation + lower_violation)


@wp.kernel
def fill_jacobian_ee_component(
    jacobian: wp.array3d(dtype=wp.float32),  # (num_envs, num_residuals_per_env, dof)
    q_grad: wp.array(dtype=wp.float32),      # (total_joints,)
    dof_per_env: int,
    ee_idx: int,
    ee_component: int,  # 0, 1, or 2 for x, y, z
):
    env_idx = wp.tid()
    
    # Start index for this environment's joints
    start_joint = env_idx * dof_per_env
    
    # Residual index for this EE component
    residual_idx = ee_idx * 3 + ee_component
    
    # Fill the jacobian row for this environment's EE residual
    for j in range(dof_per_env):
        jacobian[env_idx, residual_idx, j] = q_grad[start_joint + j]


@wp.kernel
def fill_jacobian_joint_limits(
    jacobian: wp.array3d(dtype=wp.float32),  # (num_envs, num_residuals_per_env, dof)
    q_grad: wp.array(dtype=wp.float32),      # (total_joints,)
    dof_per_env: int,
    num_ees: int,
):
    global_joint_idx = wp.tid()
    
    # Calculate indices
    env_idx = global_joint_idx / dof_per_env
    joint_idx_in_env = global_joint_idx % dof_per_env
    
    # Joint limit residual index within the environment
    residual_idx_in_env = 3 * num_ees + joint_idx_in_env
    
    # Fill the diagonal entry
    jacobian[env_idx, residual_idx_in_env, joint_idx_in_env] = q_grad[global_joint_idx]



class LevenbergMarquardtIKSolver:
    def __init__(self, model, num_envs, ee_link_indices, ee_link_offsets, num_links, num_ees, damping=1e-4, joint_limit_weight=0.1):
        self.model = model
        self.num_envs = num_envs
        self.ee_link_indices = wp.array(ee_link_indices, dtype=int)
        self.ee_link_offsets = wp.array(ee_link_offsets, dtype=wp.vec3)
        self.num_links = num_links
        self.num_ees = num_ees
        self.damping = damping
        self.joint_limit_weight = joint_limit_weight
        self.dof = len(model.joint_q) // num_envs # dof per environment
        
        self.total_joints = len(model.joint_q)
        self.num_residuals_per_env = 3 * num_ees + self.dof # 3 per EE pos + joint limits
        self.num_residuals = self.num_residuals_per_env * self.num_envs
        
        self.state = self.model.state()
        self.residuals = wp.zeros((self.num_envs, self.num_residuals_per_env), dtype=wp.float32, requires_grad=True)

        damping_diag_np = np.full(self.dof, self.damping, dtype=np.float32)
        self.damping_diag_wp = wp.array(damping_diag_np, dtype=wp.float32)

        # Pre-allocate jacobian in block-diagonal form
        self.jacobian = wp.zeros((self.num_envs, self.num_residuals_per_env, self.dof), dtype=wp.float32)

        self.tape = wp.Tape()

        # Pre-allocate e arrays for EE components
        self.e_arrays_ee = []
        for ee_idx in range(num_ees):
            for ee_component in range(3):  # x, y, z
                e = np.zeros((self.num_envs, self.num_residuals_per_env), dtype=np.float32)
                for env_idx in range(self.num_envs):
                    e[env_idx, ee_idx * 3 + ee_component] = 1.0
                self.e_arrays_ee.append(wp.array(e.flatten(), dtype=wp.float32))
        
        # Pre-allocate e array for joint limits
        e_limits = np.zeros((self.num_envs, self.num_residuals_per_env), dtype=wp.float32)
        for env_idx in range(self.num_envs):
            for joint_idx_in_env in range(self.dof):
                e_limits[env_idx, 3 * num_ees + joint_idx_in_env] = 1.0
        self.e_array_limits = wp.array(e_limits.flatten(), dtype=wp.float32)
        
        # Pre-allocate arrays for solve_tile
        self.error_norm_squared = wp.zeros(1, dtype=wp.float32)
        self.delta_q_per_env = wp.zeros((self.num_envs, self.dof), dtype=wp.float32)
        self.residuals_3d = wp.zeros((self.num_envs, self.num_residuals_per_env, 1), dtype=wp.float32)
        
        # Add CUDA graph support
        self.use_cuda_graph = wp.get_device().is_cuda
        self.graph = None
        self.target_pos_for_graph = wp.zeros((num_envs * num_ees,), dtype=wp.vec3)  # Persistent array for graph

    def _solve_iteration(self, target_pos, step_size=1.0):
        """Single iteration of the solver for CUDA graph capture."""
        # Compute residuals and jacobian
        residuals_per_env = self.compute_residuals(target_pos)
        jacobian_per_env = self.compute_jacobian(target_pos)
        
        # Compute error norm using warp kernel
        self.error_norm_squared.zero_()
        wp.launch(
            compute_ee_error_norm_squared,
            dim=self.num_envs,
            inputs=[residuals_per_env, self.error_norm_squared, self.num_ees]
        )
        
        # Reshape residuals for tile operations
        residuals_flat = residuals_per_env.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)
        
        # Zero out delta_q
        self.delta_q_per_env.zero_()
        
        # Solve using tiles
        wp.launch_tiled(
            solve_normal_equations_tiled,
            dim=[self.num_envs],
            inputs=[
                jacobian_per_env,
                self.residuals_3d,
                self.damping_diag_wp,
                self.delta_q_per_env
            ],
            block_dim=TILE_THREADS
        )
        
        # Update joint positions using kernel
        wp.launch(
            update_joint_positions,
            dim=self.total_joints,
            inputs=[
                self.model.joint_q,
                self.delta_q_per_env,
                step_size,
                self.dof
            ]
        )

    def compute_residuals(self, target_pos):
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        # Zero out residuals
        self.residuals.zero_()
        
        # Compute EE position residuals
        wp.launch(
            compute_ee_position_residuals,
            dim=self.num_envs * self.num_ees,
            inputs=[
                self.state.body_q,
                target_pos,
                self.num_links,
                self.ee_link_indices,
                self.ee_link_offsets,
                self.residuals,
                self.num_ees,
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
                self.joint_limit_weight,
                self.num_ees,
            ],
        )
        
        return self.residuals

    def compute_jacobian(self, target_pos):
        """
        Compute Jacobian matrix in block-diagonal form.
        Returns shape (num_envs, num_residuals_per_env, dof).
        """
        # Zero out jacobian for fresh computation
        self.jacobian.zero_()
        
        with self.tape:
            residuals_2d = self.compute_residuals(target_pos)
            current_residuals_wp = residuals_2d.flatten()
        
        # Compute EE position residuals for all EEs
        for ee_global_idx in range(self.num_ees * 3):
            ee_idx = ee_global_idx // 3
            ee_component = ee_global_idx % 3
            
            self.tape.backward(grads={current_residuals_wp: self.e_arrays_ee[ee_global_idx]})
            q_grad = self.tape.gradients[self.model.joint_q]
            
            # Fill all environments for this component in parallel
            wp.launch(
                fill_jacobian_ee_component,
                dim=self.num_envs,
                inputs=[
                    self.jacobian,
                    q_grad,
                    self.dof,
                    ee_idx,
                    ee_component,
                ],
            )
            
            self.tape.zero()
        
        # Compute ALL joint limit residuals in one pass
        self.tape.backward(grads={current_residuals_wp: self.e_array_limits})
        q_grad_limits = self.tape.gradients[self.model.joint_q]
        
        # Fill all joint limit entries in parallel
        wp.launch(
            fill_jacobian_joint_limits,
            dim=self.total_joints,
            inputs=[
                self.jacobian,
                q_grad_limits,
                self.dof,
                self.num_ees,
            ],
        )
    
        return self.jacobian

    def solve_tile(self, target_pos, iters=10, step_size=1.0):
        # Copy target positions to persistent array for graph
        wp.copy(self.target_pos_for_graph, target_pos)
        
        # Capture graph on first use
        if self.use_cuda_graph and self.graph is None:
            with wp.ScopedCapture() as capture:
                self._solve_iteration(self.target_pos_for_graph, step_size=step_size)
            self.graph = capture.graph
        
        for i in range(iters):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._solve_iteration(self.target_pos_for_graph, step_size=step_size)
        
###########################################################################
# Gizmo Manager
###########################################################################

class Gizmo:
    def __init__(self, name, global_target_id, axis_vector, color, highlight_color, ee_type):
        self.name = name
        self.global_target_id = global_target_id
        self.axis_vector = axis_vector
        self.original_color = color
        self.highlight_color = highlight_color
        self.color = color
        self.ee_type = ee_type  # 0=left_hand, 1=right_hand, 2=left_foot, 3=right_foot
        
        self.collision_radius = 0.0
        self.collision_half_height = 0.0
        self.offset_from_target = 0.0
        self.current_center = np.zeros(3, dtype=np.float32)
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        self.shaft_name = f"{name}_shaft"
        self.head_name = f"{name}_head"
        self.shaft_radius_factor = 0.35
        self.shaft_length_factor = 0.7
        self.head_radius_factor = 0.7
        self.head_length_factor = 0.3


class GizmoManager:
    def __init__(self, renderer, scale_factor=1.0, num_ees=4):
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.num_ees = num_ees
        self.gizmos = {}
        self.target_positions = {}
        self._highlighted_gizmo = None
        self._drag_axis_active = False
        self._drag_axis_points = None
        
        self.target_radius = 0.02
        self.gizmo_coll_radius = 0.15 * scale_factor
        self.gizmo_coll_half_height = 0.6 * scale_factor
        self.gizmo_offset_from_target = 0.05 * scale_factor
        
        # Define color schemes for each end-effector type
        self.ee_color_schemes = [
            # Left hand - Red-based
            {'X': {'color': (1.0, 0.0, 0.0), 'highlight': (1.0, 0.5, 0.5)},
             'Y': {'color': (0.8, 0.0, 0.2), 'highlight': (1.0, 0.4, 0.6)},
             'Z': {'color': (0.6, 0.0, 0.4), 'highlight': (0.8, 0.3, 0.7)}},
            # Right hand - Blue-based
            {'X': {'color': (0.0, 0.0, 1.0), 'highlight': (0.5, 0.5, 1.0)},
             'Y': {'color': (0.0, 0.2, 0.8), 'highlight': (0.4, 0.6, 1.0)},
             'Z': {'color': (0.0, 0.4, 0.6), 'highlight': (0.3, 0.7, 0.8)}},
            # Left foot - Green-based
            {'X': {'color': (0.0, 1.0, 0.0), 'highlight': (0.5, 1.0, 0.5)},
             'Y': {'color': (0.2, 0.8, 0.0), 'highlight': (0.6, 1.0, 0.4)},
             'Z': {'color': (0.4, 0.6, 0.0), 'highlight': (0.7, 0.8, 0.3)}},
            # Right foot - Yellow/Purple-based
            {'X': {'color': (1.0, 1.0, 0.0), 'highlight': (1.0, 1.0, 0.5)},
             'Y': {'color': (0.8, 0.0, 0.8), 'highlight': (1.0, 0.5, 1.0)},
             'Z': {'color': (0.6, 0.0, 0.6), 'highlight': (0.8, 0.4, 0.8)}},
            # Head - Cyan-based
            {'X': {'color': (0.0, 1.0, 1.0), 'highlight': (0.5, 1.0, 1.0)},
            'Y': {'color': (0.0, 0.8, 0.8), 'highlight': (0.4, 1.0, 1.0)},
            'Z': {'color': (0.0, 0.6, 0.6), 'highlight': (0.3, 0.8, 0.8)}},
            # Pelvis - Magenta-based
            {'X': {'color': (1.0, 0.0, 1.0), 'highlight': (1.0, 0.5, 1.0)},
            'Y': {'color': (0.8, 0.0, 0.8), 'highlight': (1.0, 0.4, 1.0)},
            'Z': {'color': (0.6, 0.0, 0.6), 'highlight': (0.8, 0.3, 0.8)}},
        ]
        
        self.renderer.render_line_strip(
            name="drag_axis_visualization",
            vertices=[],
            color=(1.0, 1.0, 0.0, 0.8),
            radius=0.015 * scale_factor
        )
    
    def create_target_gizmos(self, global_target_id, position, ee_type):
        self.target_positions[global_target_id] = np.copy(position)
        
        gizmo_axes = {
            'X': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'Y': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'Z': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        
        color_scheme = self.ee_color_schemes[ee_type]
        
        created_gizmos = []
        for axis_name, axis_vector in gizmo_axes.items():
            gizmo_name = f"GizmoArrowTarget{global_target_id}_{axis_name}"
            gizmo = Gizmo(
                gizmo_name, 
                global_target_id, 
                axis_vector, 
                color_scheme[axis_name]['color'], 
                color_scheme[axis_name]['highlight'],
                ee_type
            )
            
            gizmo.collision_radius = self.gizmo_coll_radius
            gizmo.collision_half_height = self.gizmo_coll_half_height
            gizmo.offset_from_target = self.gizmo_offset_from_target
            
            self.gizmos[gizmo_name] = gizmo
            created_gizmos.append(gizmo)
        
        self._update_gizmo_transforms(global_target_id, initial_setup=True)
        return created_gizmos
    
    def update_target_position(self, global_target_id, new_position):
        self.target_positions[global_target_id] = np.copy(new_position)
        self._update_gizmo_transforms(global_target_id)
    
    def highlight_gizmo(self, gizmo_name, highlight=True):
        if highlight:
            if self._highlighted_gizmo and self._highlighted_gizmo != gizmo_name:
                self._update_gizmo_color(self._highlighted_gizmo, highlighted=False)
            self._highlighted_gizmo = gizmo_name
            self._update_gizmo_color(gizmo_name, highlighted=True)
        else:
            if self._highlighted_gizmo == gizmo_name:
                self._highlighted_gizmo = None
                self._update_gizmo_color(gizmo_name, highlighted=False)
    
    def show_drag_axis(self, axis_origin, axis_direction):
        far_length = self.renderer.camera_far_plane * 2
        p1 = axis_origin - axis_direction * far_length
        p2 = axis_origin + axis_direction * far_length
        self._drag_axis_points = [p1.tolist(), p2.tolist()]
        self._drag_axis_active = True
        self._render_drag_axis()
    
    def hide_drag_axis(self):
        self._drag_axis_active = False
        self._drag_axis_points = None
        self._render_drag_axis()
    
    def get_gizmo(self, gizmo_name):
        return self.gizmos.get(gizmo_name)
    
    def get_all_gizmos(self):
        return list(self.gizmos.values())
    
    def get_target_position(self, global_target_id):
        return self.target_positions[global_target_id]
    
    def get_gizmo_collision_geometry(self, gizmo):
        center = gizmo.current_center
        orientation = gizmo.current_orientation
        half_height = gizmo.collision_half_height
        
        local_half_vec = np.array([0.0, half_height, 0.0], dtype=np.float32)
        rotated_half_vec = rotate_vector_by_quaternion(local_half_vec, orientation)
        return center - rotated_half_vec, center + rotated_half_vec, gizmo.collision_radius
    
    def _update_gizmo_transforms(self, global_target_id, initial_setup=False):
        target_pos = self.target_positions[global_target_id]
        
        for gizmo in self.gizmos.values():
            if gizmo.global_target_id == global_target_id:
                gizmo.current_center = target_pos + (self.target_radius + gizmo.offset_from_target + gizmo.collision_half_height) * gizmo.axis_vector
                gizmo.current_orientation = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), gizmo.axis_vector)
                
                shaft_visual_radius = gizmo.collision_radius * gizmo.shaft_radius_factor
                shaft_visual_full_length = (2 * gizmo.collision_half_height) * gizmo.shaft_length_factor
                shaft_visual_half_height = shaft_visual_full_length / 2.0
                shaft_pos = target_pos + (self.target_radius + gizmo.offset_from_target + shaft_visual_half_height) * gizmo.axis_vector
                
                if initial_setup:
                    self.renderer.render_capsule(
                        name=gizmo.shaft_name,
                        pos=tuple(shaft_pos), rot=tuple(gizmo.current_orientation),
                        radius=shaft_visual_radius, half_height=shaft_visual_half_height,
                        color=gizmo.color, up_axis=1
                    )
                else:
                    self.renderer.update_shape_instance(
                        name=gizmo.shaft_name, pos=tuple(shaft_pos), rot=tuple(gizmo.current_orientation),
                        color1=gizmo.color, color2=gizmo.color
                    )
                
                head_visual_radius = gizmo.collision_radius * gizmo.head_radius_factor
                head_visual_full_length = (2 * gizmo.collision_half_height) * gizmo.head_length_factor
                head_visual_half_height = head_visual_full_length / 2.0
                head_pos = target_pos + (self.target_radius + gizmo.offset_from_target + shaft_visual_full_length + head_visual_half_height) * gizmo.axis_vector
                
                if initial_setup:
                    self.renderer.render_capsule(
                        name=gizmo.head_name,
                        pos=tuple(head_pos), rot=tuple(gizmo.current_orientation),
                        radius=head_visual_radius, half_height=head_visual_half_height,
                        color=gizmo.color, up_axis=1
                    )
                else:
                    self.renderer.update_shape_instance(
                        name=gizmo.head_name, pos=tuple(head_pos), rot=tuple(gizmo.current_orientation),
                        color1=gizmo.color, color2=gizmo.color
                    )
    
    def _update_gizmo_color(self, gizmo_name, highlighted):
        gizmo = self.gizmos[gizmo_name]
        gizmo.color = gizmo.highlight_color if highlighted else gizmo.original_color
        global_target_id = gizmo.global_target_id
        self._update_gizmo_transforms(global_target_id)
    
    def _render_drag_axis(self):
        vertices = self._drag_axis_points if self._drag_axis_active else []
        self.renderer.render_line_strip(
            name="drag_axis_visualization",
            vertices=vertices,
            color=(1.0, 1.0, 0.0, 0.8),
            radius=0.015 * self.scale_factor
        )


###########################################################################
# Interaction Handler
###########################################################################

class InteractionHandler:
    def __init__(self, renderer, gizmo_manager, on_drag_callback, num_ees=4):
        self.renderer = renderer
        self.gizmo_manager = gizmo_manager
        self.on_drag_callback = on_drag_callback
        self.num_ees = num_ees
        
        self.dragging_gizmo_name = None
        self.drag_global_target_id = None
        self.drag_axis_vector = None
        self.drag_start_position = None
        self.drag_initial_t = None
    
    def on_mouse_press(self, x, y, button, modifiers):
        if button != pyglet.window.mouse.LEFT:
            return
            
        ray_origin, ray_dir = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return
            
        clicked_gizmo = self._find_clicked_gizmo(ray_origin, ray_dir)
        if clicked_gizmo:
            self.dragging_gizmo_name = clicked_gizmo.name
            self.drag_global_target_id = clicked_gizmo.global_target_id
            self.drag_axis_vector = np.copy(clicked_gizmo.axis_vector)
            self.drag_start_position = np.copy(self.gizmo_manager.get_target_position(self.drag_global_target_id))
            
            self.drag_initial_t = closest_point_on_line_to_ray(
                self.drag_start_position,
                self.drag_axis_vector,
                ray_origin, ray_dir
            )
            
            if self.drag_initial_t is None:
                self._cancel_drag()
                return
                
            self.gizmo_manager.highlight_gizmo(clicked_gizmo.name, highlight=True)
            self.gizmo_manager.show_drag_axis(self.drag_start_position, self.drag_axis_vector)
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.dragging_gizmo_name or not (buttons & pyglet.window.mouse.LEFT):
            return
            
        ray_origin, ray_dir = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return
            
        current_t = closest_point_on_line_to_ray(
            self.drag_start_position,
            self.drag_axis_vector,
            ray_origin, ray_dir
        )
        
        if current_t is not None and self.drag_initial_t is not None:
            delta_t = current_t - self.drag_initial_t
            movement = delta_t * self.drag_axis_vector
            new_position = self.drag_start_position + movement
            
            self.gizmo_manager.update_target_position(self.drag_global_target_id, new_position)
            self.gizmo_manager.show_drag_axis(new_position, self.drag_axis_vector)
            
            if self.on_drag_callback:
                self.on_drag_callback(self.drag_global_target_id, new_position)
    
    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT and self.dragging_gizmo_name:
            self.gizmo_manager.highlight_gizmo(self.dragging_gizmo_name, highlight=False)
            self.gizmo_manager.hide_drag_axis()
            self._cancel_drag()
    
    def _cancel_drag(self):
        self.dragging_gizmo_name = None
        self.drag_global_target_id = None
        self.drag_axis_vector = None
        self.drag_start_position = None
        self.drag_initial_t = None
    
    def _cast_ray_from_screen(self, x, y):
        screen_width = self.renderer.screen_width
        screen_height = self.renderer.screen_height
        if screen_width == 0 or screen_height == 0:
            return None, None
            
        ndc_x = (2.0 * x) / screen_width - 1.0
        ndc_y = (2.0 * y) / screen_height - 1.0
        ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        
        try:
            proj_matrix = self.renderer._projection_matrix.reshape(4, 4).T
            inv_proj = np.linalg.inv(proj_matrix)
            view_matrix = self.renderer._view_matrix.reshape(4, 4).T
            inv_view = np.linalg.inv(view_matrix)
            inv_model_matrix = np.eye(4, dtype=np.float32)
            if hasattr(self.renderer, '_inv_model_matrix') and self.renderer._inv_model_matrix is not None:
                inv_model_matrix = self.renderer._inv_model_matrix.reshape(4, 4).T
        except Exception:
            return None, None
            
        ray_eye_homo = inv_proj @ ray_clip
        if abs(ray_eye_homo[3]) < 1e-9:
            return None, None
        ray_eye_ndc = ray_eye_homo[:3] / ray_eye_homo[3]
        ray_eye_dir = np.array([ray_eye_ndc[0], ray_eye_ndc[1], -1.0, 0.0], dtype=np.float32)
        
        ray_world_dir_homo = inv_view @ ray_eye_dir
        ray_world_dir = ray_world_dir_homo[:3]
        
        ray_world_unscaled_homo = inv_model_matrix @ np.append(ray_world_dir, 0.0)
        ray_world_unscaled = ray_world_unscaled_homo[:3]
        
        norm = np.linalg.norm(ray_world_unscaled)
        if norm < 1e-6:
            return None, None
            
        ray_dir = ray_world_unscaled / norm
        ray_origin = np.array([
            self.renderer.camera_pos.x,
            self.renderer.camera_pos.y,
            self.renderer.camera_pos.z
        ], dtype=np.float32)
        
        return ray_origin, ray_dir
    
    def _find_clicked_gizmo(self, ray_origin, ray_dir):
        min_t = float('inf')
        clicked_gizmo = None
        
        for gizmo in self.gizmo_manager.get_all_gizmos():
            p_start, p_end, radius = self.gizmo_manager.get_gizmo_collision_geometry(gizmo)
            t = ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, radius)
            if 0 < t < min_t:
                min_t = t
                clicked_gizmo = gizmo
                
        return clicked_gizmo