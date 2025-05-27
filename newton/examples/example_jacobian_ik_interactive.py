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

from ik_utils import LevenbergMarquardtIKSolver, GizmoManager, InteractionHandler



###########################################################################
# Main Example
###########################################################################

class Example:
    def __init__(self, stage_path="example_jacobian_ik_interactive.usd", num_envs=10):
        self.num_envs = num_envs
        self.frame_dt = 1.0 / 60
        self.sim_time = 0.0
        
        self.model, self.num_links, self.ee_link_index, self.ee_link_offset, self.dof = self._build_model(num_envs)
        
        self.target_positions = self._initialize_targets(num_envs)
        
        self.ik_solver = LevenbergMarquardtIKSolver(
            self.model, 
            self.num_envs, 
            self.ee_link_index, 
            self.ee_link_offset, 
            self.num_links,
            joint_limit_weight=0.1
        )
        
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=1.0)
        else:
            self.renderer = None
            
        if self.renderer:
            self.gizmo_manager = GizmoManager(self.renderer, scale_factor=0.2)
            self.interaction_handler = InteractionHandler(
                self.renderer,
                self.gizmo_manager,
                on_drag_callback=self._on_target_dragged
            )
            
            for i in range(num_envs):
                self.gizmo_manager.create_target_gizmos(i, self.target_positions[i])
            
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_press)
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_drag)
            self.renderer.window.push_handlers(self.interaction_handler.on_mouse_release)
    
    def _build_model(self, num_envs):
        rng = np.random.default_rng(42)
        
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
        
        num_links = len(articulation_builder.joint_type)
        ee_link_index = num_links - 1
        ee_link_offset = wp.vec3(0.0, 0.0, 1.0)
        dof = len(articulation_builder.joint_q)
        
        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 0.0, 2.0))
        
        for i in range(num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(positions[i], wp.quat_identity()))
            builder.joint_q[-3:] = rng.uniform(-0.5, 0.5, size=3)
        
        model = builder.finalize(requires_grad=True)
        model.ground = False
        
        return model, num_links, ee_link_index, ee_link_offset, dof
    
    def _initialize_targets(self, num_envs):
        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 0.0, 2.0))
        target_origin = []
        for i in range(num_envs):
            target_origin.append((positions[i][0]+0.12, positions[i][1], positions[i][2]))
        return np.array(target_origin)
    
    def _on_target_dragged(self, target_id, new_position):
        self.target_positions[target_id] = new_position
        target_pos_wp = wp.array(self.target_positions, dtype=wp.vec3)
        self.ik_solver.solve_tile(target_pos_wp, max_iters=1, tolerance=1e-3)
    
    def _get_ee_positions(self):
        ee_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        for i in range(self.num_envs):
            body_tf = self.ik_solver.state.body_q.numpy()[i * self.num_links + self.ee_link_index]
            ee_pos[i] = wp.transform_point(
                wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                self.ee_link_offset
            )
        return ee_pos
    
    def render(self):
        if self.renderer is None:
            return
            
        self.renderer.begin_frame(self.sim_time)
        
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.ik_solver.state)
        self.renderer.render(self.ik_solver.state)
        
        self.renderer.render_points("targets", self.target_positions, radius=0.05, colors=(1.0, 0.65, 0.0))
        self.renderer.render_points("ee_pos", self._get_ee_positions(), radius=0.05, colors=(0.2, 0.8, 0.2))
        
        self.renderer.end_frame()
    
    def run(self):
        if self.renderer is None:
            return
            
        target_pos_wp = wp.array(self.target_positions, dtype=wp.vec3)
        self.ik_solver.solve_tile(target_pos_wp)
        
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
        default="example_jacobian_ik_interactive.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")
    
    args = parser.parse_known_args()[0]
    
    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)
        example.run()
