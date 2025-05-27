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
from enum import Enum
import newton
from newton.solvers.solver_featherstone import jcalc_motion

class JacobianMode(Enum):
    AUTODIFF = "autodiff"
    ANALYTIC = "analytic"
    MIXED = "mixed"

class IKObjective:
    """Base class for IK objectives"""
    
    def residual_dim(self) -> int:
        """Return the number of residuals this objective contributes"""
        raise NotImplementedError
    
    def compute_residuals(self, state, model, residuals, start_idx):
        """Compute residuals for this objective"""
        raise NotImplementedError
    
    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        """Fill in the Jacobian entries for this objective using autodiff"""
        raise NotImplementedError
    
    def supports_analytic(self) -> bool:
        """Whether this objective has an analytic Jacobian implementation"""
        return False
    
    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        """Fill in the Jacobian entries for this objective analytically"""
        pass

@wp.kernel
def compute_position_residuals_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.array(dtype=wp.vec3),
    num_links: int,
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()
    
    # Get end-effector position
    body_tf = body_q[env_idx * num_links + link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)
    
    # Compute error - now indexing into target_pos array
    error = target_pos[env_idx] - ee_pos
    
    # Write residuals
    residuals[env_idx, start_idx + 0] = error[0]
    residuals[env_idx, start_idx + 1] = error[1]
    residuals[env_idx, start_idx + 2] = error[2]

@wp.kernel
def fill_position_jacobian_component(
    jacobian: wp.array3d(dtype=wp.float32),
    q_grad: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    component: int,  # 0, 1, or 2 for x, y, z
):
    env_idx = wp.tid()
    
    # Start index for this environment's joints
    start_joint = env_idx * coords_per_env
    
    # Residual index for this component
    residual_idx = start_idx + component
    
    # Fill the jacobian row
    for j in range(coords_per_env):
        jacobian[env_idx, residual_idx, j] = q_grad[start_joint + j]

@wp.kernel
def update_target_position_kernel(
    target_array: wp.array(dtype=wp.vec3),
    env_idx: int,
    new_position: wp.vec3,
):
    target_array[env_idx] = new_position

@wp.kernel
def compute_motion_subspace_kernel(
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),  # Pass actual joint velocities
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    """Compute motion subspace (screw axes) for all joints"""
    tid = wp.tid()
    
    type = joint_type[tid]
    parent = joint_parent[tid]
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    
    X_pj = joint_X_p[tid]
    
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_pj
    
    # compute motion subspace
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]
    
    jcalc_motion(
        type,
        joint_axis,
        axis_start,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_q,
        joint_qd,  # Use actual velocities (doesn't matter for S computation)
        q_start,
        qd_start,
        joint_S_s,
    )

@wp.kernel
def compute_position_jacobian_analytic_kernel(
    link_index: int,
    link_offset: wp.vec3,
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    start_idx: int,
    num_links_per_env: int,
    coords_per_env: int,
    dof_per_env: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Compute position Jacobian for a single end-effector"""
    env_idx = wp.tid()

    # Find which articulation this environment belongs to
    env_coord_start = env_idx * coords_per_env
    articulation_idx = int(0)

    # Walk through articulations to find which one contains our coord
    for i in range(len(articulation_start) - 1):
        joint_start_i = articulation_start[i]
        joint_end_i = articulation_start[i + 1]
        articulation_coord_start_i = joint_q_start[joint_start_i]
        articulation_coord_end_i = joint_q_start[joint_end_i]
        
        if env_coord_start >= articulation_coord_start_i and env_coord_start < articulation_coord_end_i:
            articulation_idx = i
            break

    # Now get the articulation info
    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_start[articulation_idx + 1]
    articulation_coord_start = joint_q_start[joint_start]
    
    # For multi-robot case, calculate actual body index
    body_idx = env_idx * num_links_per_env + link_index
    
    # Get end-effector position in world frame
    ee_transform = body_q[body_idx]
    ee_offset_world = wp.quat_rotate(wp.transform_get_rotation(ee_transform), link_offset)
    ee_pos_world = wp.transform_get_translation(ee_transform) + ee_offset_world
    
    # Walk up the kinematic chain
    current_body = body_idx
    
    while current_body >= 0:
        # Find which joint moves this body
        joint_idx = int(-1)
        for j in range(joint_start, joint_end):
            if joint_child[j] == current_body:
                joint_idx = j
                break
        
        if joint_idx == -1:
            break
        
        # Get coordinate range for this joint
        joint_coord_start = joint_q_start[joint_idx]
        joint_coord_end = joint_q_start[joint_idx + 1]
        
        # Check if this is a free joint
        if joint_type[joint_idx] == wp.int32(4):  # JOINT_FREE = 4
            # For free joints, coordinates are [x, y, z, qx, qy, qz, qw]
            
            # Translation derivatives (first 3 coords)
            for i in range(3):
                col = joint_coord_start + i - env_idx * coords_per_env
                jacobian[env_idx, start_idx + i, col] = -1.0
                jacobian[env_idx, start_idx + (i+1)%3, col] = 0.0
                jacobian[env_idx, start_idx + (i+2)%3, col] = 0.0
            
            # Quaternion derivatives (coords 3-6)
            # Get current quaternion
            qx = joint_q[joint_coord_start + 3]
            qy = joint_q[joint_coord_start + 4]
            qz = joint_q[joint_coord_start + 5]
            qw = joint_q[joint_coord_start + 6]

            # Get vector from joint to end-effector in local frame
            joint_transform = body_q[current_body]
            joint_quat = wp.transform_get_rotation(joint_transform)
            joint_pos = wp.transform_get_translation(joint_transform)

            # Vector from joint to EE in world frame
            r_world = ee_pos_world - joint_pos

            # Transform to local frame (inverse rotate)
            r_local = wp.quat_rotate(wp.quat_inverse(joint_quat), r_world)

            # Compute quaternion derivatives
            # These come from d/dq (q * r_local * q*)
            rx = r_local[0]
            ry = r_local[1]
            rz = r_local[2]

            # ∂/∂qx
            col = joint_coord_start + 3 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qy * ry + qz * rz)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qy * rx - 2.0 * qx * ry - qw * rz)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (qz * rx + qw * ry - 2.0 * qx * rz)

            # ∂/∂qy  
            col = joint_coord_start + 4 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qx * ry - 2.0 * qy * rx + qw * rz)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qx * rx + qz * rz)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (-qw * rx + qz * ry - 2.0 * qy * rz)

            # ∂/∂qz
            col = joint_coord_start + 5 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qx * rz - qw * ry - 2.0 * qz * rx)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qw * rx + qy * rz - 2.0 * qz * ry)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (qx * rx + qy * ry)

            # ∂/∂qw - special case for identity quaternion
            col = joint_coord_start + 6 - env_idx * coords_per_env
            if wp.abs(qw - 1.0) < 1e-6 and wp.abs(qx) < 1e-6 and wp.abs(qy) < 1e-6 and wp.abs(qz) < 1e-6:
                # Matching autodiff at identity
                jacobian[env_idx, start_idx + 0, col] = -2.0 * rx
                jacobian[env_idx, start_idx + 1, col] = -2.0 * ry  
                jacobian[env_idx, start_idx + 2, col] = -4.0 * rz
            else:
                # General case
                jacobian[env_idx, start_idx + 0, col] = 2.0 * (qz * ry - qy * rz)
                jacobian[env_idx, start_idx + 1, col] = 2.0 * (-qz * rx + qx * rz)
                jacobian[env_idx, start_idx + 2, col] = 2.0 * (qy * rx - qx * ry)
        else:
            # For other joint types, we still use velocity-based Jacobian
            # We assume coords are 1:1 with dofs for this joint
            # (false for some joint types)
            joint_dof_start = joint_qd_start[joint_idx]
            for coord_i in range(joint_coord_end - joint_coord_start):
                # Get corresponding DOF (for most joints, coord == dof)
                dof = joint_dof_start + coord_i
                coord = joint_coord_start + coord_i
                
                # Get motion vector
                S = joint_S_s[dof]
                
                # Extract angular and linear parts
                omega = wp.vec3(S[0], S[1], S[2])
                v_origin = wp.vec3(S[3], S[4], S[5])
                
                # Linear velocity at end-effector = v_origin + omega × ee_position
                v_ee = v_origin + wp.cross(omega, ee_pos_world)
                
                # Column index based on coordinates
                col = coord - env_idx * coords_per_env
                
                # Fill Jacobian (negative for residual)
                jacobian[env_idx, start_idx + 0, col] = -v_ee[0]
                jacobian[env_idx, start_idx + 1, col] = -v_ee[1]
                jacobian[env_idx, start_idx + 2, col] = -v_ee[2]
        
        # Move to parent body
        current_body = joint_parent[joint_idx]

class PositionObjective(IKObjective):
    """Position objective for a single end-effector"""
    
    def __init__(self, link_index, link_offset, target_positions, num_links, num_envs, total_residuals, residual_offset):
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.num_links = num_links
        self.num_envs = num_envs
        
        # Pre-allocate e arrays for jacobian computation
        self.e_arrays = []
        for component in range(3):
            e = np.zeros((num_envs, total_residuals), dtype=np.float32)
            for env_idx in range(num_envs):
                e[env_idx, residual_offset + component] = 1.0
            self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32))
        
        # Pre-allocate space for motion subspace (will be filled by solver if using analytic)
        self.joint_S_s = None

    def allocate_motion_subspace(self, total_dofs):
        """Allocate space for motion subspace if not already done"""
        if self.joint_S_s is None:
            self.joint_S_s = wp.zeros(total_dofs, dtype=wp.spatial_vector)

    def supports_analytic(self) -> bool:
        """Position objectives support analytic Jacobian"""
        return True

    def set_target_position(self, env_idx, new_position):
        """Update the target position for a specific environment"""
        wp.launch(
            update_target_position_kernel,
            dim=1,
            inputs=[
                self.target_positions,
                env_idx,
                new_position
            ]
        )
        
    def residual_dim(self):
        return 3  # x, y, z
    
    def compute_residuals(self, state, model, residuals, start_idx):
        """Compute position residuals for all environments"""
        wp.launch(
            compute_position_residuals_kernel,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.target_positions,
                self.num_links,
                self.link_index,
                self.link_offset,
                start_idx,
            ],
            outputs=[residuals]
        )
    
    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        """Compute Jacobian using autodiff"""
        # Compute gradients for each component
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[model.joint_q]
            coords_per_env = model.joint_coord_count // self.num_envs
            
            wp.launch(
                fill_position_jacobian_component,
                dim=self.num_envs,
                inputs=[
                    jacobian,
                    q_grad,
                    coords_per_env,
                    start_idx,
                    component,
                ],
            )
            
            tape.zero()

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        """Compute Jacobian analytically using spatial motion subspace"""
        # Ensure arrays are allocated
        total_dofs = len(model.joint_qd)
        self.allocate_motion_subspace(total_dofs)
        
        # Compute motion subspace for all joints
        num_joints = model.joint_count
        wp.launch(
            compute_motion_subspace_kernel,
            dim=num_joints,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_q,
                model.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_axis_dim,
                state.body_q,
                model.joint_X_p,
                self.joint_S_s,
            ]
        )

        coords_per_env = model.joint_coord_count // self.num_envs
        dof_per_env = model.joint_dof_count // self.num_envs
        wp.launch(
            compute_position_jacobian_analytic_kernel,
            dim=self.num_envs,
            inputs=[
                self.link_index,
                self.link_offset,
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_type,
                model.joint_q,
                self.joint_S_s,
                state.body_q,
                start_idx,
                self.num_links,
                coords_per_env,
                dof_per_env
            ],
            outputs=[jacobian]
        )

@wp.kernel
def compute_joint_limit_residuals_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    coords_per_env: int,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()
    
    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env
    
    q = joint_q[global_joint_idx]
    lower = joint_limit_lower[joint_idx_in_env]
    upper = joint_limit_upper[joint_idx_in_env]
    
    upper_violation = wp.max(0.0, q - upper)
    lower_violation = wp.max(0.0, lower - q)
    
    residuals[env_idx, start_idx + joint_idx_in_env] = weight * (upper_violation + lower_violation)


@wp.kernel
def fill_joint_limit_jacobian(
    q_grad: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()
    
    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env
    
    residual_idx_in_env = start_idx + joint_idx_in_env
    
    jacobian[env_idx, residual_idx_in_env, joint_idx_in_env] = q_grad[global_joint_idx]


@wp.kernel
def compute_joint_limit_jacobian_analytic_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    weight: float,
    jacobian: wp.array3d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()
    
    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env
    
    q = joint_q[global_joint_idx]
    lower = joint_limit_lower[joint_idx_in_env]
    upper = joint_limit_upper[joint_idx_in_env]
    
    # Compute joint limit gradient
    # Residual = weight * (max(0, q - upper) + max(0, lower - q))
    # Gradient:
    #   - If q > upper: d/dq = weight
    #   - If q < lower: d/dq = -weight  
    #   - Otherwise: d/dq = 0
    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight
    
    # Write to jacobian
    residual_idx = start_idx + joint_idx_in_env
    jacobian[env_idx, residual_idx, joint_idx_in_env] = grad


class JointLimitObjective(IKObjective):
    """Joint limit objective to keep joints within bounds"""
    
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.1, num_envs=None, total_residuals=None, residual_offset=None):
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.weight = weight
        self.e_array = None
        self.num_envs = num_envs

        # Pre-allocate e array if dimensions are provided
        self.coords_per_env = len(joint_limit_lower) // num_envs
        if num_envs is not None and total_residuals is not None and residual_offset is not None:
            e = np.zeros((num_envs, total_residuals), dtype=np.float32)
            for env_idx in range(self.num_envs):
                for joint_idx in range(self.coords_per_env):
                    e[env_idx, residual_offset + joint_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32)
        else:
            self.e_array = None

    def supports_analytic(self) -> bool:
        """Joint limits have a simple analytic Jacobian"""
        return True
 
    def residual_dim(self):
        return self.coords_per_env
    
    def compute_residuals(self, state, model, residuals, start_idx):
        """Compute joint limit residuals"""
        wp.launch(
            compute_joint_limit_residuals_kernel,
            dim=model.joint_coord_count,
            inputs=[
                model.joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coords_per_env,
                self.weight,
                start_idx,
            ],
            outputs=[residuals]
        )
    
    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        """Compute Jacobian using autodiff"""
        tape.backward(grads={tape.outputs[0]: self.e_array})
        q_grad = tape.gradients[model.joint_q]
        
        wp.launch(
            fill_joint_limit_jacobian,
            dim=model.joint_coord_count,
            inputs=[
                q_grad,
                self.coords_per_env,
                start_idx,
            ],
            outputs=[jacobian]
        )

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        """Compute Jacobian analytically for joint limits"""
        wp.launch(
            compute_joint_limit_jacobian_analytic_kernel,
            dim=model.joint_coord_count,
            inputs=[
                model.joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coords_per_env,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian]
        )