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

from ik_objectives import IKObjective

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


def ray_box_intersect(ray_origin, ray_dir, box_center, box_extents, box_orientation):
    """Ray-OBB (Oriented Bounding Box) intersection."""
    inv_orientation = np.array([
        -box_orientation[0], -box_orientation[1], -box_orientation[2], box_orientation[3]
    ], dtype=np.float32)
    
    local_ray_origin = rotate_vector_by_quaternion(ray_origin - box_center, inv_orientation)
    local_ray_dir = rotate_vector_by_quaternion(ray_dir, inv_orientation)
    
    tmin = float('-inf')
    tmax = float('inf')
    
    for i in range(3):
        if abs(local_ray_dir[i]) < 1e-6:
            if abs(local_ray_origin[i]) > box_extents[i] / 2.0:
                return float('inf')
        else:
            t1 = (-box_extents[i] / 2.0 - local_ray_origin[i]) / local_ray_dir[i]
            t2 = (box_extents[i] / 2.0 - local_ray_origin[i]) / local_ray_dir[i]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
    
    if tmin > tmax or tmax < 0:
        return float('inf')
    
    return tmin if tmin >= 0 else tmax


###########################################################################
# Gizmo Manager
###########################################################################

class Gizmo:
    def __init__(self, name, global_target_id, axis_vector, gizmo_type='arrow'):
        self.name = name
        self.global_target_id = global_target_id
        self.axis_vector = axis_vector
        self.gizmo_type = gizmo_type  # 'arrow' or 'plane'
        
        # Colors will be set based on axis
        self.original_color = (0.7, 0.7, 0.7)
        self.highlight_color = (0.85, 0.85, 0.85)
        self.color = self.original_color
        
        self.collision_radius = 0.0
        self.collision_half_height = 0.0
        self.offset_from_target = 0.0
        self.current_center = np.zeros(3, dtype=np.float32)
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        if gizmo_type == 'arrow':
            self.capsule_name = f"{name}_capsule"
            self.capsule_radius_factor = 0.08  # Thicker capsule for visibility
            # Generous collision capsule
            self.collision_radius_factor = 0.2  # Generous radius for easy clicking
        elif gizmo_type == 'plane':
            self.visual_name = f"{name}_visual"
            self.axis1 = None  # Will be set for plane gizmos
            self.axis2 = None  # Will be set for plane gizmos
            self.normal = None  # Will be set for plane gizmos
            self.size = 0.0
            self.thickness = 0.0
            self.distance_factor = 0.0
            self.display_size = 0.0  # Visual display size


class GizmoManager:
    def __init__(self, renderer, scale_factor=1.0, num_ees=4, show_visual_aids=False):
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.num_ees = num_ees
        self.show_visual_aids = show_visual_aids  # Controls ee/target/line rendering
        self.gizmos = {}
        self.gizmo_positions = {}  # Gizmo positions
        self.robot_positions = {}  # Corresponding robot positions
        self._highlighted_gizmo = None
        self._drag_axis_active = False
        self._drag_axis_points = None
        
        self.target_radius = 0.02
        self.gizmo_coll_radius = 0.3 * scale_factor
        self.gizmo_coll_half_height = 0.5 * scale_factor
        self.gizmo_offset_from_target = 0.0  # Arrows start from center
        
        # Plane handle parameters (bigger collision, smaller display)
        self.plane_handle_size = 0.4 * scale_factor  # Collision size
        self.plane_handle_display_size = 0.3 * scale_factor  # Half size for display
        self.plane_handle_thickness = 0.02 * scale_factor
        self.plane_handle_distance_factor = 0.35  # Closer to center
        
        # Visual parameters
        self.target_point_radius = 0.02
        self.ee_point_radius = 0.02
        self.connection_line_radius = 0.005 * scale_factor
        self.target_color = (0.8, 0.8, 0.8)
        self.ee_color = (0.6, 0.6, 0.6)
        self.connection_color = (0.5, 0.5, 0.5, 0.5)
        
        # Axis colors
        self.axis_colors = {
            'X': {'color': (0.8, 0.2, 0.2), 'highlight': (1.0, 0.4, 0.4)},  # Red
            'Y': {'color': (0.2, 0.8, 0.2), 'highlight': (0.4, 1.0, 0.4)},  # Green
            'Z': {'color': (0.2, 0.2, 0.8), 'highlight': (0.4, 0.4, 1.0)},  # Blue
        }
    
    def create_target_gizmos(self, global_target_id, robot_position, gizmo_position):
        """Create gizmos at specified position for a robot target at robot_position"""
        self.gizmo_positions[global_target_id] = np.copy(gizmo_position)
        self.robot_positions[global_target_id] = np.copy(robot_position)
        
        gizmo_axes = {
            'X': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'Y': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'Z': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        
        created_gizmos = []
        
        # Create arrow gizmos
        for axis_name, axis_vector in gizmo_axes.items():
            gizmo_name = f"GizmoArrowTarget{global_target_id}_{axis_name}"
            gizmo = Gizmo(
                gizmo_name, 
                global_target_id, 
                axis_vector, 
                gizmo_type='arrow'
            )
            
            # Set axis-specific colors
            gizmo.original_color = self.axis_colors[axis_name]['color']
            gizmo.highlight_color = self.axis_colors[axis_name]['highlight']
            gizmo.color = gizmo.original_color
            
            gizmo.collision_radius = self.gizmo_coll_radius
            gizmo.collision_half_height = self.gizmo_coll_half_height
            gizmo.offset_from_target = self.gizmo_offset_from_target
            
            self.gizmos[gizmo_name] = gizmo
            created_gizmos.append(gizmo)
        
        # Create plane gizmos
        plane_configs = {
            'XY': {
                'normal': np.array([0.0, 0.0, 1.0], dtype=np.float32),
                'axis1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'axis2': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'stable_axis': 'Z',  # This plane holds Z stable
            },
            'XZ': {
                'normal': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'axis1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'axis2': np.array([0.0, 0.0, 1.0], dtype=np.float32),
                'stable_axis': 'Y',  # This plane holds Y stable
            },
            'YZ': {
                'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'axis1': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'axis2': np.array([0.0, 0.0, 1.0], dtype=np.float32),
                'stable_axis': 'X',  # This plane holds X stable
            }
        }
        
        for plane_name, config in plane_configs.items():
            gizmo_name = f"GizmoPlaneTarget{global_target_id}_{plane_name}"
            gizmo = Gizmo(
                gizmo_name,
                global_target_id,
                config['normal'],  # Use normal as axis_vector
                gizmo_type='plane'
            )
            
            # Set color based on the stable axis
            stable_axis = config['stable_axis']
            gizmo.original_color = self.axis_colors[stable_axis]['color']
            gizmo.highlight_color = self.axis_colors[stable_axis]['highlight']
            gizmo.color = gizmo.original_color
            
            gizmo.axis1 = config['axis1']
            gizmo.axis2 = config['axis2']
            gizmo.normal = config['normal']
            gizmo.size = self.plane_handle_size  # Collision size
            gizmo.display_size = self.plane_handle_display_size  # Visual size
            gizmo.thickness = self.plane_handle_thickness
            gizmo.distance_factor = self.plane_handle_distance_factor
            gizmo.collision_half_height = self.gizmo_coll_half_height
            gizmo.offset_from_target = self.gizmo_offset_from_target
            
            self.gizmos[gizmo_name] = gizmo
            created_gizmos.append(gizmo)
        
        self._update_gizmo_transforms(global_target_id, initial_setup=True)
        return created_gizmos
    
    def update_gizmo_position(self, global_target_id, new_gizmo_position):
        """Update gizmo position (keeping robot position offset)"""
        offset = self.gizmo_positions[global_target_id] - self.robot_positions[global_target_id]
        self.gizmo_positions[global_target_id] = np.copy(new_gizmo_position)
        self.robot_positions[global_target_id] = new_gizmo_position - offset
        self._update_gizmo_transforms(global_target_id)
    
    def update_robot_position(self, global_target_id, new_robot_position):
        """Update robot position (keeping gizmo offset)"""
        offset = self.gizmo_positions[global_target_id] - self.robot_positions[global_target_id]
        self.robot_positions[global_target_id] = np.copy(new_robot_position)
        self.gizmo_positions[global_target_id] = new_robot_position + offset
        self._update_gizmo_transforms(global_target_id)
    
    def update_ee_position(self, global_target_id, ee_position):
        """Update the actual end-effector position for rendering"""
        # This is stored temporarily for rendering, not persisted
        self._current_ee_positions = getattr(self, '_current_ee_positions', {})
        self._current_ee_positions[global_target_id] = np.copy(ee_position)
    
    def render_all_targets(self):
        """Render all visual elements (spheres, lines, etc)"""
        if self.show_visual_aids:
            current_ee_positions = getattr(self, '_current_ee_positions', {})
            
            for global_id in self.gizmo_positions:
                # Render target sphere at gizmo position
                self.renderer.render_points(
                    f"target_{global_id}", 
                    [self.gizmo_positions[global_id]], 
                    radius=self.target_point_radius,
                    colors=self.target_color
                )
                
                # Render end-effector sphere if we have the position
                if global_id in current_ee_positions:
                    self.renderer.render_points(
                        f"ee_pos_{global_id}", 
                        [current_ee_positions[global_id]], 
                        radius=self.ee_point_radius,
                        colors=self.ee_color
                    )
                    
                    # Render connection line
                    vertices = [
                        current_ee_positions[global_id].tolist(),
                        self.gizmo_positions[global_id].tolist()
                    ]
                    self.renderer.render_line_strip(
                        name=f"connection_line_{global_id}",
                        vertices=vertices,
                        color=self.connection_color,
                        radius=self.connection_line_radius
                    )
    
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
    
    def show_drag_axis(self, axis_origin, axis_direction, plane_mode=False, axis2_direction=None):
        far_length = self.renderer.camera_far_plane * 2
        
        if plane_mode and axis2_direction is not None:
            # Show cross pattern for plane dragging
            p1 = axis_origin - axis_direction * far_length
            p2 = axis_origin + axis_direction * far_length
            p3 = axis_origin - axis2_direction * far_length
            p4 = axis_origin + axis2_direction * far_length
            self._drag_axis_points = [p1.tolist(), p2.tolist(), p2.tolist(), p3.tolist(), p3.tolist(), p4.tolist()]
        else:
            # Single axis line
            p1 = axis_origin - axis_direction * far_length
            p2 = axis_origin + axis_direction * far_length
            self._drag_axis_points = [p1.tolist(), p2.tolist()]
        
        self._drag_axis_active = True
        # Immediately render to ensure it's visible
        self.renderer.render_line_strip(
            name="drag_axis_visualization",
            vertices=self._drag_axis_points,
            color=(1.0, 1.0, 0.0, 0.8),
            radius=0.015 * self.scale_factor
        )
    
    def hide_drag_axis(self):
        self._drag_axis_active = False
        self._drag_axis_points = None
        # Try to delete the line strip if the renderer supports it
        if hasattr(self.renderer, 'delete_shape'):
            self.renderer.delete_shape("drag_axis_visualization")
        elif hasattr(self.renderer, 'remove_shape'):
            self.renderer.remove_shape("drag_axis_visualization")
        else:
            # Fallback: render with zero radius and transparent
            self.renderer.render_line_strip(
                name="drag_axis_visualization",
                vertices=[[0, 0, 0], [0, 0, 0]],
                color=(1.0, 1.0, 0.0, 0.0),
                radius=0.0
            )
    
    def get_gizmo(self, gizmo_name):
        return self.gizmos.get(gizmo_name)
    
    def get_all_gizmos(self):
        return list(self.gizmos.values())
    
    def get_gizmo_position(self, global_target_id):
        return self.gizmo_positions[global_target_id]
    
    def get_robot_position(self, global_target_id):
        return self.robot_positions[global_target_id]
    
    def get_gizmo_collision_geometry(self, gizmo):
        if gizmo.gizmo_type == 'arrow':
            # Return simplified collision capsule for arrow
            gizmo_pos = self.gizmo_positions[gizmo.global_target_id]
            # Capsule starts from center and extends along axis
            start = gizmo_pos
            end = gizmo_pos + gizmo.axis_vector * (self.gizmo_coll_half_height * 2)
            radius = self.scale_factor * gizmo.collision_radius_factor
            return start, end, radius
        elif gizmo.gizmo_type == 'plane':
            # Return plane info for collision detection (use full size for collision)
            return gizmo.current_center, gizmo.current_orientation, gizmo.size
    
    def _get_plane_transform(self, gizmo):
        """Calculate position and orientation for plane handle box."""
        gizmo_pos = self.gizmo_positions[gizmo.global_target_id]
        
        # Calculate position based on arrow lengths
        total_arrow_length = self.gizmo_coll_half_height * 2
        plane_distance = total_arrow_length * gizmo.distance_factor
        
        # Position planes along the diagonal of the two axes
        diagonal_dir = (gizmo.axis1 + gizmo.axis2) / np.sqrt(2.0)
        
        # Also add a small offset along the plane normal to ensure it doesn't intersect with arrows
        normal_offset = gizmo.normal * self.plane_handle_size * 0.2
        
        plane_center = gizmo_pos + diagonal_dir * 2*plane_distance + normal_offset
        
        # Simple direct quaternion assignment based on plane type
        if 'XY' in gizmo.name:
            # XY plane - no rotation needed, already aligned with Z pointing forward
            orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        elif 'XZ' in gizmo.name:
            # XZ plane - rotate 90 degrees around X axis to make Z point up
            # This makes the thin dimension (local Z) align with world Y
            orientation = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)
        elif 'YZ' in gizmo.name:
            # YZ plane - rotate 90 degrees around Y axis to make Z point right
            # This makes the thin dimension (local Z) align with world X  
            orientation = np.array([0.0, 0.70710678, 0.0, 0.70710678], dtype=np.float32)
        else:
            # Fallback to identity
            orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        return plane_center, orientation
    
    def _update_gizmo_transforms(self, global_target_id, initial_setup=False):
        gizmo_pos = self.gizmo_positions[global_target_id]
        
        for gizmo in self.gizmos.values():
            if gizmo.global_target_id == global_target_id:
                if gizmo.gizmo_type == 'arrow':
                    # Capsule starts from gizmo center
                    gizmo.current_center = gizmo_pos + gizmo.collision_half_height * gizmo.axis_vector
                    gizmo.current_orientation = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), gizmo.axis_vector)
                    
                    # Calculate capsule parameters
                    half_height = self.gizmo_coll_half_height
                    radius = self.scale_factor * gizmo.capsule_radius_factor
                    
                    # Capsule position is at its center
                    capsule_pos = gizmo_pos + half_height * gizmo.axis_vector
                    
                    if initial_setup:
                        self.renderer.render_capsule(
                            name=gizmo.capsule_name,
                            pos=tuple(capsule_pos),
                            rot=tuple(gizmo.current_orientation),
                            half_height=half_height,
                            radius=radius,
                            up_axis=1,
                            color=gizmo.color,
                        )
                    self.renderer.update_shape_instance(
                        name=gizmo.capsule_name,
                        pos=tuple(capsule_pos),
                        rot=tuple(gizmo.current_orientation),
                        color1=gizmo.color,
                        color2=gizmo.color
                    )
                
                elif gizmo.gizmo_type == 'plane':
                    center_pos, orientation = self._get_plane_transform(gizmo)
                    gizmo.current_center = center_pos
                    gizmo.current_orientation = orientation
                    
                    # Use display_size for rendering, but size for collision
                    # render_box expects half-extents, so divide display_size by 2
                    if initial_setup:
                        self.renderer.render_box(
                            name=gizmo.visual_name,
                            pos=tuple(center_pos),
                            rot=tuple(orientation),
                            extents=(gizmo.display_size/2, gizmo.display_size/2, gizmo.thickness/2),
                            color=gizmo.color,
                        )
                    self.renderer.update_shape_instance(
                        name=gizmo.visual_name,
                        pos=tuple(center_pos),
                        rot=tuple(orientation),
                        color1=gizmo.color,
                        color2=gizmo.color
                    )
    
    def _update_gizmo_color(self, gizmo_name, highlighted):
        gizmo = self.gizmos[gizmo_name]
        gizmo.color = gizmo.highlight_color if highlighted else gizmo.original_color
        global_target_id = gizmo.global_target_id
        self._update_gizmo_transforms(global_target_id)


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
        
        # Plane dragging state
        self.drag_plane_normal = None
        self.drag_plane_axis1 = None
        self.drag_plane_axis2 = None
        self.drag_initial_plane_point = None
    
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
            self.drag_start_position = np.copy(self.gizmo_manager.get_gizmo_position(self.drag_global_target_id))
            
            if clicked_gizmo.gizmo_type == 'arrow':
                self.drag_axis_vector = np.copy(clicked_gizmo.axis_vector)
                
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
            
            elif clicked_gizmo.gizmo_type == 'plane':
                self.drag_plane_normal = np.copy(clicked_gizmo.normal)
                self.drag_plane_axis1 = np.copy(clicked_gizmo.axis1)
                self.drag_plane_axis2 = np.copy(clicked_gizmo.axis2)
                
                # Calculate initial intersection point on plane
                plane_d = np.dot(self.drag_plane_normal, self.drag_start_position)
                denom = np.dot(ray_dir, self.drag_plane_normal)
                
                if abs(denom) < 1e-6:
                    self._cancel_drag()
                    return
                
                t = (plane_d - np.dot(ray_origin, self.drag_plane_normal)) / denom
                if t < 0:
                    self._cancel_drag()
                    return
                
                self.drag_initial_plane_point = ray_origin + t * ray_dir
                
                self.gizmo_manager.highlight_gizmo(clicked_gizmo.name, highlight=True)
                self.gizmo_manager.show_drag_axis(self.drag_start_position, self.drag_plane_axis1, 
                                                 plane_mode=True, axis2_direction=self.drag_plane_axis2)
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.dragging_gizmo_name or not (buttons & pyglet.window.mouse.LEFT):
            return
            
        ray_origin, ray_dir = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return
        
        clicked_gizmo = self.gizmo_manager.get_gizmo(self.dragging_gizmo_name)
        if not clicked_gizmo:
            return
            
        if clicked_gizmo.gizmo_type == 'arrow':
            current_t = closest_point_on_line_to_ray(
                self.drag_start_position,
                self.drag_axis_vector,
                ray_origin, ray_dir
            )
            
            if current_t is not None and self.drag_initial_t is not None:
                delta_t = current_t - self.drag_initial_t
                movement = delta_t * self.drag_axis_vector
                new_position = self.drag_start_position + movement
                
                self.gizmo_manager.update_gizmo_position(self.drag_global_target_id, new_position)
                self.gizmo_manager.show_drag_axis(new_position, self.drag_axis_vector)
                
                if self.on_drag_callback:
                    # Pass the robot position to the callback
                    robot_position = self.gizmo_manager.get_robot_position(self.drag_global_target_id)
                    self.on_drag_callback(self.drag_global_target_id, robot_position)
        
        elif clicked_gizmo.gizmo_type == 'plane':
            # Intersect ray with plane
            plane_d = np.dot(self.drag_plane_normal, self.drag_start_position)
            denom = np.dot(ray_dir, self.drag_plane_normal)
            
            if abs(denom) > 1e-6:
                t = (plane_d - np.dot(ray_origin, self.drag_plane_normal)) / denom
                if t > 0:
                    current_plane_point = ray_origin + t * ray_dir
                    
                    # Calculate movement on plane
                    plane_movement = current_plane_point - self.drag_initial_plane_point
                    
                    # Project movement onto the two plane axes
                    movement_axis1 = np.dot(plane_movement, self.drag_plane_axis1) * self.drag_plane_axis1
                    movement_axis2 = np.dot(plane_movement, self.drag_plane_axis2) * self.drag_plane_axis2
                    total_movement = movement_axis1 + movement_axis2
                    
                    new_position = self.drag_start_position + total_movement
                    
                    self.gizmo_manager.update_gizmo_position(self.drag_global_target_id, new_position)
                    self.gizmo_manager.show_drag_axis(new_position, self.drag_plane_axis1,
                                                     plane_mode=True, axis2_direction=self.drag_plane_axis2)
                    
                    if self.on_drag_callback:
                        # Pass the robot position to the callback
                        robot_position = self.gizmo_manager.get_robot_position(self.drag_global_target_id)
                        self.on_drag_callback(self.drag_global_target_id, robot_position)
    
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
        self.drag_plane_normal = None
        self.drag_plane_axis1 = None
        self.drag_plane_axis2 = None
        self.drag_initial_plane_point = None
    
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
            if gizmo.gizmo_type == 'arrow':
                # Use simplified capsule collision
                p_start, p_end, radius = self.gizmo_manager.get_gizmo_collision_geometry(gizmo)
                t = ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, radius)
                if 0 < t < min_t:
                    min_t = t
                    clicked_gizmo = gizmo
            elif gizmo.gizmo_type == 'plane':
                center, orientation, size = self.gizmo_manager.get_gizmo_collision_geometry(gizmo)
                t = ray_box_intersect(ray_origin, ray_dir, center, 
                                    np.array([size, size, gizmo.thickness]), orientation)
                if 0 < t < min_t:
                    min_t = t
                    clicked_gizmo = gizmo
                
        return clicked_gizmo