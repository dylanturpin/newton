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
import newton.examples
import newton.utils

from ik_objectives import PositionObjective, JointLimitObjective
from ik_solver import create_ik_solver, JacobianMode
from gizmo_interaction import GizmoManager, InteractionHandler

class Example:
    def __init__(self, stage_path="example_h1_ik_interactive.usd", num_envs=4):
        self.num_envs = num_envs
        self.frame_dt = 1.0 / 60
        self.sim_time = 0.0
        
        # Define end-effectors (only hands and feet)
        self.ee_names = ["left_hand", "right_hand", "left_foot", "right_foot"]
        self.num_ees = len(self.ee_names)
        
        # Define gizmo offsets for each end-effector
        self.gizmo_offset_distance = 0.3
        self.gizmo_offsets = [
            np.array([0.0, 0.0, -self.gizmo_offset_distance], dtype=np.float32),  # left_hand - offset backward
            np.array([0.0, 0.0, self.gizmo_offset_distance], dtype=np.float32),   # right_hand - offset forward
            np.array([0.0, 0.0, -self.gizmo_offset_distance], dtype=np.float32),  # left_foot - offset backward
            np.array([0.0, 0.0, self.gizmo_offset_distance], dtype=np.float32),   # right_foot - offset forward
        ]
        
        # Build model
        self.model, self.num_links, self.ee_link_indices, self.ee_link_offsets, self.coords = self._build_model(num_envs)
        
        # Initialize targets from current FK
        self.target_positions = self._initialize_targets(num_envs)
        
        # Create objectives
        self.objectives = []
        self.position_objectives = []  # Keep references for updating targets

        # First calculate total residuals
        total_residuals = self.num_ees * 3 + self.coords  # position residuals + joint limit residuals

        # Create shared target arrays for all position objectives
        self.target_arrays = []
        for ee_idx in range(self.num_ees):
            # Create array with targets for all environments for this end-effector
            targets = np.array([self.target_positions[env_idx, ee_idx] for env_idx in range(num_envs)])
            target_array = wp.array(targets, dtype=wp.vec3)
            self.target_arrays.append(target_array)

        # Create position objectives
        for ee_idx in range(self.num_ees):
            pos_obj = PositionObjective(
                link_index=self.ee_link_indices[ee_idx],
                link_offset=self.ee_link_offsets[ee_idx],
                target_positions=self.target_arrays[ee_idx],
                num_links=self.num_links,
                num_envs=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=ee_idx * 3
            )
            self.objectives.append(pos_obj)
            self.position_objectives.append(pos_obj)

        # Create joint limit objective
        joint_limit_obj = JointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=0.1,
            num_envs=self.num_envs,
            total_residuals=total_residuals,
            residual_offset=self.num_ees * 3
        )
        self.objectives.append(joint_limit_obj)
        
        # Create solver using factory function
        self.ik_solver = create_ik_solver(
            model=self.model,
            num_envs=self.num_envs,
            objectives=self.objectives,
            damping=1.0,
            jacobian_mode=JacobianMode.ANALYTIC
        )
        
        # Setup renderer and interaction
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=1.0)
        else:
            self.renderer = None
            
        if self.renderer:
            self.gizmo_manager = GizmoManager(self.renderer, scale_factor=0.15, num_ees=self.num_ees, show_visual_aids=False)
            self.interaction_handler = InteractionHandler(
                self.renderer,
                self.gizmo_manager,
                on_drag_callback=self._on_target_dragged,
                num_ees=self.num_ees
            )
            
            # Create gizmos for each environment with offsets
            for env_idx in range(num_envs):
                for ee_idx in range(self.num_ees):
                    global_id = env_idx * self.num_ees + ee_idx
                    robot_position = self.target_positions[env_idx, ee_idx]
                    gizmo_position = robot_position + self.gizmo_offsets[ee_idx]
                    
                    self.gizmo_manager.create_target_gizmos(
                        global_id,
                        robot_position,
                        gizmo_position
                    )
            
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_press)
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_drag)
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_release)
    
    def _build_model(self, num_envs):
        """Build the H1 robot model"""
        rng = np.random.default_rng(42)
        
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1
        
        # Load H1 robot with hands from MJCF
        newton.utils.parse_mjcf(
            newton.examples.get_asset("h1_description/mjcf/h1_with_hand.xml"),
            articulation_builder,
            floating=False,
            armature_scale=1.0,
            scale=1.0,
        )

        # Set initial joint positions for a reasonable pose
        initial_joint_positions = [
            0.0,   # left_hip_yaw
            0.0,   # left_hip_roll  
            -0.3,  # left_hip_pitch
            0.6,   # left_knee
            -0.3,  # left_ankle
            0.0,   # right_hip_yaw
            0.0,   # right_hip_roll
            -0.3,  # right_hip_pitch
            0.6,   # right_knee
            -0.3,  # right_ankle
            0.0,   # torso
            0.0,   # left_shoulder_pitch
            0.0,   # left_shoulder_roll
            0.0,   # left_shoulder_yaw
            -0.5,  # left_elbow
            0.0,   # right_shoulder_pitch
            -0.3,  # right_shoulder_roll
            0.0,   # right_shoulder_yaw
            -0.8,  # right_elbow
        ]
        
        # Joint mapping from initial_joint_positions to model joints (after 7 floating base coords)
        joint_mapping = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 35, 36, 37, 38]
        
        for i, (joint_idx, value) in enumerate(zip(joint_mapping, initial_joint_positions)):
            articulation_builder.joint_q[joint_idx-7] = value
        
        # Set hand joints to neutral positions
        articulation_builder.joint_q[22-7] = 0.0  # left_hand_joint
        articulation_builder.joint_q[39-7] = 0.0  # right_hand_joint
        
        # Set all finger joints to neutral positions
        for i in range(23, 35):  # left hand finger joints
            articulation_builder.joint_q[i-7] = 0.0
        for i in range(40, 52):  # right hand finger joints
            articulation_builder.joint_q[i-7] = 0.0
        
        builder = newton.ModelBuilder()
        
        # Get robot properties
        num_links = len(articulation_builder.body_q)
        
        # Identify end-effector body indices (only hands and feet)
        ee_link_indices = [
            16,  # left_hand_link
            33,  # right_hand_link (after 12 left finger bodies)
            5,   # left_ankle_link
            10,  # right_ankle_link
        ]

        # Define offsets
        hand_offset = wp.vec3(0.0, 0.0, 0.0)
        foot_offset = wp.vec3(0.0, 0.0, 0.0)
        ee_link_offsets = [hand_offset, hand_offset, foot_offset, foot_offset]
        
        coords = len(articulation_builder.joint_q)
        
        # Position environments
        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 0.0, 2.0))
        
        for i in range(num_envs):
            builder.add_builder(
                articulation_builder, 
                xform=wp.transform(positions[i], wp.quat_identity())
            )
        
        model = builder.finalize(requires_grad=True)
        model.ground = True
        
        return model, num_links, ee_link_indices, ee_link_offsets, coords

    def _initialize_targets(self, num_envs):
        """Initialize targets at current end-effector positions"""
        state = self.model.state()
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state, None)
        
        target_positions = np.zeros((num_envs, self.num_ees, 3), dtype=np.float32)
        for i in range(num_envs):
            for ee_idx in range(self.num_ees):
                body_tf = state.body_q.numpy()[i * self.num_links + self.ee_link_indices[ee_idx]]
                ee_pos = wp.transform_point(
                    wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                    self.ee_link_offsets[ee_idx]
                )
                target_positions[i, ee_idx] = ee_pos
        
        return target_positions
    
    def _on_target_dragged(self, global_target_id, new_position):
        """Handle target dragging"""
        env_idx = global_target_id // self.num_ees
        ee_idx = global_target_id % self.num_ees
        self.target_positions[env_idx, ee_idx] = new_position
        
        # Update the position objective's target
        self.position_objectives[ee_idx].set_target_position(
            env_idx,
            wp.vec3(new_position[0], new_position[1], new_position[2])
        )
        
        # Solve
        with wp.ScopedTimer("solve", print=False):
            self.ik_solver.solve(iterations=20)
    
    def _get_ee_positions(self):
        """Get current end-effector positions"""
        ee_pos = np.zeros((self.num_envs, self.num_ees, 3), dtype=np.float32)
        for i in range(self.num_envs):
            for ee_idx in range(self.num_ees):
                body_tf = self.ik_solver.state.body_q.numpy()[i * self.num_links + self.ee_link_indices[ee_idx]]
                ee_pos[i, ee_idx] = wp.transform_point(
                    wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                    self.ee_link_offsets[ee_idx]
                )
        return ee_pos
    
    def render(self):
        """Render the scene"""
        if self.renderer is None:
            return
            
        self.renderer.begin_frame(self.sim_time)
        
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.ik_solver.state)
        self.renderer.render(self.ik_solver.state)
        
        # Update end-effector positions in gizmo manager
        ee_positions = self._get_ee_positions()
        for env_idx in range(self.num_envs):
            for ee_idx in range(self.num_ees):
                global_id = env_idx * self.num_ees + ee_idx
                self.gizmo_manager.update_ee_position(global_id, ee_positions[env_idx, ee_idx])
        
        # Let gizmo manager handle all rendering
        self.gizmo_manager.render_all_targets()
        
        self.renderer.end_frame()
    
    def run(self):
        """Main loop"""
        if self.renderer is None:
            return
        
        # Initial solve
        self.ik_solver.solve(iterations=10)
        
        while self.renderer.is_running():
            self.sim_time += self.frame_dt
            self.render()
            self.renderer.update()
        
        self.renderer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_h1_ik_interactive.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")
    
    args = parser.parse_known_args()[0]
    
    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)
        example.run()