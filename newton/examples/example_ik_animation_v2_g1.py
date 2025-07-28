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

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.sim.ik as ik
import newton.utils
from newton.sim import eval_fk
from newton.utils.gizmo import GizmoSystem

import xml.etree.ElementTree as ET
import math
from pyglet.window import key
from newton.core.spatial import quat_to_rpy 

# -------------------------------------------------------------------------
# Utility classes
# -------------------------------------------------------------------------


class FrameAlignedHandler:
    """Buffers GUI events and forwards the *latest* one once per render frame."""

    def __init__(self, handler: Callable, consume_ret: Callable[..., bool] | None = None):
        self._handler = handler
        self._consume_ret = consume_ret or (lambda *_: False)
        self._pending: tuple | None = None

    def __call__(self, *args):
        self._pending = args
        return self._consume_ret(*args)

    def flush(self):
        if self._pending is None:
            return
        self._handler(*self._pending)
        self._pending = None


# -------------------------------------------------------------------------
# Example
# -------------------------------------------------------------------------


class Example:
    """Interactive inverse-kinematics playground for a batch of g1 robots."""

    DEFAULT_POSE_FILE = 'D:/src/newton/animation/Unitree_Default.xml'
    ANIMATION_FILE = 'D:/src/newton/animation/Unitree_Getup.xml'

    POS_END_EFFECTOR = (["left_shoulder", 0.001], 
                          ["left_elbow", 0.001], 
                          ["left_wrist", 0.01], 
                          ["right_shoulder", 0.001], 
                          ["right_elbow", 0.001], 
                          ["right_wrist", 0.01],
                          ["left_hip", 0.001], 
                          ["left_knee", 0.001], 
                          ["left_ankle", 0.01], 
                          ["right_hip", 0.001], 
                          ["right_knee", 0.001], 
                          ["right_ankle", 0.01], 
                          ["torso", 0.001])
    
    ROT_END_EFFECTOR = (["left_wrist", 0.01],
                          ["right_wrist", 0.01], 
                          ["left_ankle", 0.01],  
                          ["right_ankle", 0.01],
                          ["torso", 0.01]) 
#                          "waist")


    MATCHING_JOINT_NAMES = ("Ctrl_LeftArm", "Ctrl_LeftForeArm", "Ctrl_LeftHand", 
                            "Ctrl_RightArm", "Ctrl_RightForeArm", "Ctrl_RightHand",
                            "Ctrl_LeftUpLeg", "Ctrl_LeftLeg", "Ctrl_LeftFoot",
                            "Ctrl_RightUpLeg", "Ctrl_RightLeg", "Ctrl_RightFoot",
                            "Ctrl_Spine")        

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        stage_path: str | None = "example_g1_ik_interactive.usd",
        num_envs: int = 2,
        tie_targets: bool = True,
        ik_iters: int = 20,
    ):
        self.stage_path = stage_path
        self.num_envs = num_envs
        self.tie_targets = tie_targets
        self.ik_iters = ik_iters

        # timings ------------------------------------------------------
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # model(s) -----------------------------------------------------
        (
            self.model,
            self.num_links,
            self.pos_ee_link_indices,
            self.pos_ee_link_offsets,
            self.rot_ee_link_indices,
            self.rot_ee_link_offsets,
            self.num_dofs,
            self.env_offsets,
        ) = self._build_model(num_envs)

        #throw error if the number of end effectors does not match the number of link indices
        if (len(self.pos_ee_link_indices) != len(self.POS_END_EFFECTOR)):
            raise ValueError(f"pos_ee_link_indices length {len(self.pos_ee_link_indices)} does not match POS_END_EFFECTOR length {len(self.POS_END_EFFECTOR)}")
        if (len(self.rot_ee_link_indices) != len(self.ROT_END_EFFECTOR)):
            raise ValueError(f"rot_ee_link_indices length {len(self.rot_ee_link_indices)} does not match ROT_END_EFFECTOR length {len(self.ROT_END_EFFECTOR)}")

        # dedicated 1-env model for IK
        self.singleton_model, *_ = self._build_model(1)

        # simulation state --------------------------------------------
        self.state = self.model.state()
        self.joint_q = wp.array(self.model.joint_q, shape=(num_envs, self.singleton_model.joint_coord_count), copy=True)

        # target buffers ----------------------------------------------
        self.target_positions, self.target_rotations = self._initialize_targets()
        (
            self.position_objectives,
            self.rotation_objectives,
            total_residuals,
        ) = self._create_objectives()

        # joint limits -------------------------------------------------
        joint_limit_objective = ik.JointLimitObjective(
            joint_limit_lower=self.singleton_model.joint_limit_lower,
            joint_limit_upper=self.singleton_model.joint_limit_upper,
            n_problems=num_envs,
            total_residuals=total_residuals,
            residual_offset=(len(self.POS_END_EFFECTOR) + len(self.ROT_END_EFFECTOR)) * 3,
            weight=1.0,
        )

        # IK solver ----------------------------------------------------
        self.ik_solver = ik.IKSolver(
            model=self.singleton_model,
            joint_q=self.joint_q,
            objectives=self.position_objectives + self.rotation_objectives + [joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.JacobianMode.ANALYTIC,
        )

        self.rot_mapping = None

        # renderer & gizmos -------------------------------------------
        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=1.0)
            self._setup_gizmos()

        # warm-up + CUDA graph ----------------------------------------
        self.use_cuda_graph = wp.get_device().is_cuda
        self.ik_solver.solve(iterations=ik_iters)  # JIT & cache
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.ik_solver.solve(iterations=ik_iters)
                wp.copy(self.model.joint_q, self.joint_q.flatten())
            self.graph = capture.graph

        # initialize animation data
        self.current_frame = 0
        self.frame_count = 0
        self.animation_fps = 30

        self.delta_q_data = self.calculate_delta_q(self.DEFAULT_POSE_FILE)
        self.anim_data = self.read_joint_transforms_xml(self.ANIMATION_FILE)
        
        self.frame_count = len(self.anim_data['Ctrl_Hips'])
        self.play_animation = False

    @staticmethod
    def read_joint_transforms_xml(filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        joint_data = {}

        # converter from Y up Z forward to Z up and X forward
        q1 = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.radians(90.0))
        q2 = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.radians(90.0))
        converter = q1 * q2
        converter_inverse = wp.quat_inverse(converter)

        def print_axis_angle(quat, name):
            axis, angle = wp.quat_to_axis_angle(quat)
            print(f"{name} axis: {axis}, angle: {math.degrees(angle)}")

        # test rotation conversion    
        def test_rotation_conversion(origina_axis, name):
            quat = wp.quat_from_axis_angle(origina_axis, math.radians(30.0))
            quat2 = converter * quat * converter_inverse
            axis, angle = wp.quat_to_axis_angle(quat2)
            print(f"{name} quat: {axis}, {math.degrees(angle)}")

        test_rotation_conversion(wp.vec3(0, 0, 1), "z axis")
        test_rotation_conversion(wp.vec3(0, 1, 0), "y axis")
        test_rotation_conversion(wp.vec3(1, 0, 0), "x axis")
        
        for joint_elem in root.findall('joint'):
            joint_name = joint_elem.attrib['name']
            joint_data[joint_name] = []
            for key_elem in joint_elem.findall('key'):
                key_id = int(key_elem.attrib['id'])
                t_text = key_elem.find('t').text
                q_text = key_elem.find('q').text

                # Parse position and quaternion
                position = [float(x)/100.0 for x in t_text.split(',')]
                quaternion = [float(x) for x in q_text.split(',')]

                wpos = wp.quat_rotate(converter, position)

                wquat = converter * wp.quat(quaternion) * converter_inverse
                #wquat = wp.quat(quaternion[0], quaternion[2], quaternion[1], quaternion[3])
                print(f"joint name: {joint_name} original position: {position}, converted position: {wpos}")
                print(f"joint name: {joint_name} original quaternion: {quaternion}, converted quaternion: {wquat}")
                print_axis_angle(wp.quat(quaternion), f"{joint_name} original quaternion")
                print_axis_angle(wquat, f"{joint_name} converted quaternion")
                joint_data[joint_name].append({
                    'id': key_id,
                    'position': wpos,
                    'quaternion': wquat
                })

        return joint_data

    def calculate_delta_q(self, filename):
        joint_data = self.read_joint_transforms_xml(filename)
        body_q_np = self.state.body_q.numpy()
        delta_q_data = {}
        # only look for the first frame
        for ee_idx, joint_name in enumerate(self.MATCHING_JOINT_NAMES):
            if joint_name not in joint_data:
                continue
            keyframes = joint_data[joint_name]
            if len(keyframes) > 0:
                keyframe = keyframes[0]
                maya_tf = wp.transform(keyframe['position'], wp.quat(keyframe['quaternion']))
                newton_tf = body_q_np[self.pos_ee_link_indices[ee_idx]]
                delta_tf = wp.transform_inverse(maya_tf)*newton_tf
                delta_q_data[joint_name] = wp.transform_get_rotation(delta_tf)

        return delta_q_data
    
    # -----------------------------------------------------------------
    # Model construction helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_model(num_envs: int):
        """Return (`model`, `num_links`, `ee_indices`, `ee_offsets`, `n_coords`, `env_offsets`)."""
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_mjcf(
            newton.utils.download_asset("g1_description") / "g1_29dof_rev_1_0.xml",
            articulation_builder,
            floating=True,
        )

        # initial joint angles
        for i in range(1, len(articulation_builder.joint_q)):
            lo = articulation_builder.joint_limit_lower[i-1]
            hi = articulation_builder.joint_limit_upper[i-1]
            if lo > -1e5 and hi < 1e5:
                #articulation_builder.joint_q[i] = 0.5 * (lo + hi)
                articulation_builder.joint_q[i] = 0.

        # wrap into batched ModelBuilder ------------------------------
        builder = newton.ModelBuilder()
        builder.num_rigid_contacts_per_env = 0

        env_offsets = newton.examples.compute_env_offsets(num_envs, env_offset=(-1.0, -2.0, 0.0))

        #print (f"env_offsets: {env_offsets}")
        for pos in env_offsets:
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        # left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist, left_hip, left_knee, left_ankle,right_hip, right_knee, right_ankle, 
        # torso, waist
        pos_ee_link_indices = [18, 19, 22, 25, 26, 29, 3, 4, 6, 9, 10, 12, 15]  
        pos_ee_link_offsets = [wp.vec3()] * len(pos_ee_link_indices)

        #left_wrist, right_wrist, left_ankle, right_ankle, torso, waist
        rot_ee_link_indices = [22, 29, 6, 12, 15]#, 14]  
        #rot_ee_link_indices = [22, 29, 15, 14]  
        rot_ee_link_offsets = [wp.vec3()] * len(rot_ee_link_indices)

        return (
            model,
            articulation_builder.body_count,
            pos_ee_link_indices,
            pos_ee_link_offsets,
            rot_ee_link_indices,
            rot_ee_link_offsets,
            articulation_builder.joint_dof_count,
            env_offsets,
        )

    # -----------------------------------------------------------------
    # Targets & Objectives
    # -----------------------------------------------------------------

    def _initialize_targets(self):
        """Compute initial world-frame targets from current FK."""
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

        pos_num_ees = len(self.POS_END_EFFECTOR)
        rot_num_ees = len(self.ROT_END_EFFECTOR)
        tgt_pos = np.zeros((self.num_envs, pos_num_ees, 3), dtype=np.float32)
        tgt_rot = np.zeros((self.num_envs, rot_num_ees, 4), dtype=np.float32)

        body_q_np = self.state.body_q.numpy()
        for env in range(self.num_envs):
            base = env * self.num_links
            for ee_idx, link_idx in enumerate(self.pos_ee_link_indices):
                tf = body_q_np[base + link_idx]
                world_pos = wp.transform_point(wp.transform(tf[:3], wp.quat(*tf[3:])), self.pos_ee_link_offsets[ee_idx])
                tgt_pos[env, ee_idx] = np.array(world_pos)
                print(f"environment {env} effeector {ee_idx} init target position: {tgt_pos[env, ee_idx]}")

            for ee_idx, link_idx in enumerate(self.rot_ee_link_indices):
                tf = body_q_np[base + link_idx]
                quat = tf[3:7] / np.linalg.norm(tf[3:7])
                tgt_rot[env, ee_idx] = quat
                print(f"environment {env} effeector {ee_idx} init target rotation: {tgt_rot[env, ee_idx]}")
        return tgt_pos, tgt_rot

    def _create_objectives(self):
        pos_num_ees = len(self.POS_END_EFFECTOR)
        rot_num_ees = len(self.ROT_END_EFFECTOR)
        # the last num_dofs are for joint limits
        total_residuals = ( pos_num_ees + rot_num_ees ) * 3 * 2 + self.num_dofs

        position_objectives, rotation_objectives = [], []
        self.position_target_arrays, self.rotation_target_arrays = [], []

        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(self.target_positions[:, ee_idx], dtype=wp.vec3)
            self.position_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(self.target_rotations[:, ee_idx], dtype=wp.vec4)
            self.rotation_target_arrays.append(rot_wp)

        # position objectives -----------------------------------------
        for ee_idx, (link_idx, offset) in enumerate(zip(self.pos_ee_link_indices, self.pos_ee_link_offsets)):
            obj = ik.PositionObjective(
                link_index=link_idx,
                link_offset=offset,
                target_positions=self.position_target_arrays[ee_idx],
                n_problems=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=ee_idx * 3,
                weight=1.0,
            )
            position_objectives.append(obj)

        # rotation objectives -----------------------------------------

        for ee_idx, link_idx in enumerate(self.rot_ee_link_indices):
            obj = ik.RotationObjective(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=self.rotation_target_arrays[ee_idx],
                n_problems=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=pos_num_ees * 3 + ee_idx * 3,
                weight=1.0,
            )
            rotation_objectives.append(obj)

        return position_objectives, rotation_objectives, total_residuals

    def _on_key_press(self, symbol, modifiers):
        if symbol == key.PAGEUP:
            self.current_frame = (self.current_frame + 1) % self.frame_count
            print(f"current frame: {self.current_frame}")
            self._update_animation(self.current_frame)

        elif symbol == key.PAGEDOWN:
            self.current_frame = (self.current_frame - 1) % self.frame_count
            print(f"current frame: {self.current_frame}")
            self._update_animation(self.current_frame)
            
        elif symbol == key.ENTER:
            self.play_animation = not self.play_animation
            if self.play_animation:
                print("playing animation")
            else:
                print("stopped animation")

    # -----------------------------------------------------------------
    # Gizmos & mouse interaction
    # -------------------------------------------------------------
    
    GIZMO_OFFSET_DISTANCE = 0.1
    GIZMO_OFFSETS = (
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),                
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 2*GIZMO_OFFSET_DISTANCE], dtype=np.float32),
        np.array([0.0, 0.0, 2*GIZMO_OFFSET_DISTANCE], dtype=np.float32),
    )
    def find_rotation_target_index(self, position_index):
        if self.rot_mapping is None:
            self.rot_mapping = {}
            # cache it first time
            for re_idx, (rot_name, rot_weight) in enumerate(self.ROT_END_EFFECTOR):
                for ee_idx, (name, weight) in enumerate(self.POS_END_EFFECTOR):
                    if name == rot_name:
                        self.rot_mapping[ee_idx] = re_idx
                        break
                    
        if position_index in self.rot_mapping:
            return self.rot_mapping[position_index]
        else:
            return -1
        
    def _setup_gizmos(self):
        self.gizmo_system = GizmoSystem(self.renderer, scale_factor=0.15, rotation_sensitivity=1.0)
        self.gizmo_system.set_callbacks(
            position_callback=self._on_position_dragged,
            rotation_callback=self._on_rotation_dragged,
        )

        # for now, it creates gizmos for position targets only
        # allow rotation change if it has the same name as a position target
        if self.tie_targets:
            for ee_idx in range(len(self.POS_END_EFFECTOR)):
                world_pos = self.target_positions[0, ee_idx]
                rot_idx = self.find_rotation_target_index(ee_idx)
                if rot_idx != -1:
                    self.gizmo_system.create_target(
                        ee_idx,
                        world_pos,
                        self.target_rotations[0, rot_idx],
                        self.GIZMO_OFFSETS[ee_idx],
                    )
                else:
                    self.gizmo_system.create_target(
                        ee_idx,
                        world_pos,
                        None,
                        self.GIZMO_OFFSETS[ee_idx],
                    )
        else:
            for env in range(self.num_envs):
                for ee_idx in range(len(self.POS_END_EFFECTOR)):
                    gid = env * len(self.POS_END_EFFECTOR) + ee_idx
                    world_pos = self.target_positions[env, ee_idx]
                    rot_idx = self.find_rotation_target_index(ee_idx)
                    if rot_idx != -1:                    
                        self.gizmo_system.create_target(
                            gid,
                            world_pos,
                            self.target_rotations[env, rot_idx],
                            self.GIZMO_OFFSETS[ee_idx],
                        )
                    else:
                        self.gizmo_system.create_target(
                            gid,
                            world_pos,
                            None,
                            self.GIZMO_OFFSETS[ee_idx],
                        )
        self.gizmo_system.finalize()

        # frame-aligned wrappers --------------------------------------
        self._mouse_press_handler = FrameAlignedHandler(self.gizmo_system.on_mouse_press)
        self._mouse_drag_handler = FrameAlignedHandler(
            self.gizmo_system.on_mouse_drag,
            consume_ret=lambda *_: self.gizmo_system.drag_state is not None,
        )
        self._mouse_release_handler = FrameAlignedHandler(self.gizmo_system.on_mouse_release)

        # register pyglet callbacks
        self.renderer.window.push_handlers(on_mouse_press=self._mouse_press_handler)
        self.renderer.window.push_handlers(on_mouse_drag=self._mouse_drag_handler)
        self.renderer.window.push_handlers(on_mouse_release=self._mouse_release_handler)
        self.renderer.window.push_handlers(on_key_press=self._on_key_press)

        # tied-target drag bookkeeping
        if self.tie_targets:
            self._drag_start_positions = None
            self._drag_start_rotations = None
            self._is_dragging_pos = False
            self._is_dragging_rot = False
            self._last_drag_id: int | None = None

    # snapshot helpers for tied drag ----------------------------------

    def _capture_drag_start(self):
        self._drag_start_positions = np.copy(self.target_positions)
        self._drag_start_rotations = np.copy(self.target_rotations)

    # callbacks -------------------------------------------------------

    def _on_position_dragged(self, global_id: int, new_world_pos: np.ndarray):
        env = global_id // len(self.POS_END_EFFECTOR)
        ee = global_id % len(self.POS_END_EFFECTOR)

        if self.tie_targets:
            if not self._is_dragging_pos or self._last_drag_id != global_id:
                self._capture_drag_start()
                self._is_dragging_pos = True
                self._last_drag_id = global_id

            delta = new_world_pos - self._drag_start_positions[env, ee]
            new_targets = self._drag_start_positions[:, ee] + delta
            self.target_positions[:, ee] = new_targets
            self.position_objectives[ee].set_target_positions(wp.array(new_targets, dtype=wp.vec3))
            self._is_dragging_pos = False
            print(new_targets)
        else:
            self.target_positions[env, ee] = new_world_pos
            self.position_objectives[ee].set_target_position(env, wp.vec3(*new_world_pos))
            print(new_world_pos)

        self._solve()

    def _on_rotation_dragged(self, global_id: int, new_q: np.ndarray):
        pass
        num_ees = len(self.POS_END_EFFECTOR)
        env = global_id // num_ees
        ee = global_id % num_ees

        new_q = new_q / np.linalg.norm(new_q)

        rot_idx = self.find_rotation_target_index(ee)
        if self.tie_targets:
            # start of a new drag?
            if (not self._is_dragging_rot) or (self._last_drag_id != global_id):
                self._drag_start_rotations = np.copy(self.target_rotations)
                self._is_dragging_rot = True
                self._last_drag_id = global_id

            # delta = new * conj(initial)
            q0 = self._drag_start_rotations[env, ee]
            conj = np.array([-q0[0], -q0[1], -q0[2], q0[3]], dtype=np.float32)

            delta = np.array(
                [
                    new_q[3] * conj[0] + new_q[0] * conj[3] + new_q[1] * conj[2] - new_q[2] * conj[1],
                    new_q[3] * conj[1] - new_q[0] * conj[2] + new_q[1] * conj[3] + new_q[2] * conj[0],
                    new_q[3] * conj[2] + new_q[0] * conj[1] - new_q[1] * conj[0] + new_q[2] * conj[3],
                    new_q[3] * conj[3] - new_q[0] * conj[0] - new_q[1] * conj[1] - new_q[2] * conj[2],
                ],
                dtype=np.float32,
            )
            delta /= np.linalg.norm(delta)

            # apply the same delta to every env's stored initial rotation
            initial = self._drag_start_rotations[:, ee]  # shape (num_envs, 4)
            q1, q2 = delta, initial  # aliases to match original math

            updated = np.zeros_like(initial)
            updated[:, 0] = q1[3] * q2[:, 0] + q1[0] * q2[:, 3] + q1[1] * q2[:, 2] - q1[2] * q2[:, 1]
            updated[:, 1] = q1[3] * q2[:, 1] - q1[0] * q2[:, 2] + q1[1] * q2[:, 3] + q1[2] * q2[:, 0]
            updated[:, 2] = q1[3] * q2[:, 2] + q1[0] * q2[:, 1] - q1[1] * q2[:, 0] + q1[2] * q2[:, 3]
            updated[:, 3] = q1[3] * q2[:, 3] - q1[0] * q2[:, 0] - q1[1] * q2[:, 1] - q1[2] * q2[:, 2]

            # normalise all rows (protects against numeric drift)
            updated /= np.linalg.norm(updated, axis=1, keepdims=True)

            self.target_rotations[:, rot_idx] = updated
            self.rotation_objectives[rot_idx].set_target_rotations(wp.array(updated, dtype=wp.vec4))

            self._is_dragging_rot = False
            print (updated)
        else:
            # untied mode: update only this environment
            self.target_rotations[env, rot_idx] = new_q
            self.rotation_objectives[rot_idx].set_target_rotation(env, wp.vec4(*new_q))
            axis, angle = wp.quat_to_axis_angle(wp.quat(new_q))
            print(f"new rotation: quat: {new_q}, axis: {axis}, angle: {math.degrees(angle)}")

        # re-solve IK
        self._solve()

    def _update_animation(self, current_frame):
        env_id = 0
        offset = self.env_offsets[env_id]
  
        for ee_idx, joint_name in enumerate(self.MATCHING_JOINT_NAMES):
            if joint_name not in self.anim_data:
                continue
            keyframes = self.anim_data[joint_name]
            if len(keyframes) > 0:
                keyframe = keyframes[current_frame // 3]
                key = keyframe['id']
                # use hip's position as base offset without Z
                #@todo: we should add rotation of hip as well
                base_offset = offset + self.anim_data["Ctrl_Hips"][0]['position']
                base_offset[2] = 0.0
                position = base_offset + keyframe['position']
                rotation = keyframe['quaternion'] * self.delta_q_data[joint_name]
                self.target_positions[0, ee_idx] = position
                self.position_objectives[ee_idx].set_target_position(0, wp.vec3(*position))
                gid = env_id * len(self.POS_END_EFFECTOR) + ee_idx
                self.gizmo_system.update_target_position(gid, position)

                #update rotation target if it has the same name as a position target
                rot_idx = self.find_rotation_target_index(ee_idx)
                if rot_idx != -1:
                    #quat = wp.quat(rotation)
                    #rpy = quat_to_rpy(quat)
                    print (f"updatig rotation for {self.ROT_END_EFFECTOR[rot_idx]} rotation: {rotation}")
                    self.target_rotations[0, rot_idx] = rotation 
                    self.gizmo_system.update_target_rotation(gid, rotation)
                    self.rotation_objectives[rot_idx].set_target_rotation(0, wp.vec4(*rotation))

        self._solve()

    def _advance_animation(self):
        current_frame = (self.current_frame + 1) % (self.frame_count * 3)
        self._update_animation(current_frame)
        self.current_frame = current_frame

    # -----------------------------------------------------------------
    # Solve / render / run
    # -----------------------------------------------------------------

    def _solve(self):
        #with wp.ScopedTimer("solve", synchronize=True):
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.ik_solver.solve(iterations=self.ik_iters)
            wp.copy(self.model.joint_q, self.joint_q.flatten())

    def _render_frame(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        self.renderer.render(self.state)
        self.renderer.end_frame()

    def run(self):
        # initial solve so joints match targets before first frame
        self._solve()
        while self.renderer.is_running():
            self.sim_time += self.frame_dt

            # process any pending GUI events
            self._mouse_press_handler.flush()
            self._mouse_drag_handler.flush()
            self._mouse_release_handler.flush()
            if self.play_animation:
                self._advance_animation()
            self._render_frame()

        self.renderer.close()


# -------------------------------------------------------------------------
# main()
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_g1_ik_interactive.usd",
        help="Path of the output USD.",
    )
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments.")
    parser.add_argument(
        "--tie_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie all envs together so dragging one EE moves the others.",
    )
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_envs=args.num_envs,
            tie_targets=args.tie_targets,
        )
        example.run()


if __name__ == "__main__":
    main()
