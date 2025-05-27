import numpy as np
import warp as wp
import unittest
import newton
import newton.core.articulation
import newton.utils
from ik_objectives import PositionObjective
import os


class TestPositionJacobianFloating(unittest.TestCase):
    def setUp(self):
        self.num_envs = 1  # Single robot
        
        # Build model from MJCF
        self.model, self.num_links, self.ee_link_index, self.ee_offset, self.coords = self._build_model()
        
        # Compute forward kinematics
        self.state = self.model.state()

        print(f"\nModel structure:")
        print(f"Joint child: {self.model.joint_child.numpy()}")
        print(f"Joint parent: {self.model.joint_parent.numpy()}")
        print(f"Joint type: {self.model.joint_type.numpy()}")
        print(f"Joint qd_start: {self.model.joint_qd_start.numpy()}")
        print(f"Body count: {len(self.state.body_q)}")
        print(f"Total coords: {self.coords}")

        newton.core.articulation.eval_fk(
            self.model, 
            self.model.joint_q, 
            self.model.joint_qd, 
            self.state, 
            None
        )
        
        # Get initial EE position and set target
        self.target_position = self._initialize_target()
        
        # Create position objective
        self.total_residuals = 3  # Position only (x, y, z)
        self.objective = PositionObjective(
            link_index=self.ee_link_index,
            link_offset=self.ee_offset,
            target_positions=wp.array([self.target_position], dtype=wp.vec3),
            num_links=self.num_links,
            num_envs=self.num_envs,
            total_residuals=self.total_residuals,
            residual_offset=0
        )
        
        # Override supports_analytic to return True
        self.objective.supports_analytic = lambda: True
    
    def _build_model(self):
        """Build robot model from MJCF"""
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1
        
        # Load floating robot from MJCF
        mjcf_path = os.path.join(os.path.dirname(__file__), "floating_robot.xml")
        newton.utils.parse_mjcf(
            mjcf_path,
            articulation_builder,
            floating=True,  # Already has freejoint in MJCF
            armature_scale=1.0,
            scale=1.0,
        )
        
        # Set initial joint positions
        # Free joint: 7 values (pos + quat)
        # Revolute joints: 3 values
        initial_positions = [
            0.0, 0.0, 1.0,  # base position
            0.0, 0.0, 0.0, 1.0,  # base quaternion (x,y,z,w)
            0.1,  # joint1
            0.2,  # joint2
            0.3,  # joint3
        ]
        
        for i, value in enumerate(initial_positions):
            articulation_builder.joint_q[i] = value
        
        builder = newton.ModelBuilder()
        builder.add_builder(
            articulation_builder, 
            xform=wp.transform_identity()
        )
        
        model = builder.finalize(requires_grad=True)
        model.ground = True
        
        # Get robot properties
        num_links = len(articulation_builder.body_q)
        coords = len(model.joint_q)
        
        # End-effector is the last link
        ee_link_index = num_links - 1
        ee_offset = wp.vec3(0.0, 0.0, 0.0)  # Center of ee_link
        
        return model, num_links, ee_link_index, ee_offset, coords
    
    def _initialize_target(self):
        """Initialize target from current end-effector position"""
        # Get actual EE position
        body_tf = self.state.body_q.numpy()[self.ee_link_index]
        ee_pos = wp.transform_point(
            wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
            self.ee_offset
        )
        
        # Set target slightly offset from current position
        target = ee_pos + wp.vec3(0.1, 0.05, -0.05)
        return target
    
    def test_jacobian_comparison(self):
        """Compare autodiff and analytic Jacobians"""
        # Allocate arrays
        residuals = wp.zeros((self.num_envs, self.total_residuals), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((self.num_envs, self.total_residuals, self.coords), dtype=wp.float32)
        jacobian_analytic = wp.zeros((self.num_envs, self.total_residuals, self.coords), dtype=wp.float32)
        
        # Compute residuals
        self.objective.compute_residuals(self.state, self.model, residuals, 0)
        
        # Compute Jacobian with autodiff
        tape = wp.Tape()
        with tape:
            newton.core.articulation.eval_fk(
                self.model, 
                self.model.joint_q, 
                self.model.joint_qd, 
                self.state, 
                None
            )
            residuals_compute = wp.zeros((self.num_envs, self.total_residuals), dtype=wp.float32, requires_grad=True)
            self.objective.compute_residuals(self.state, self.model, residuals_compute, 0)
            residuals_flat = residuals_compute.flatten()
        
        tape.outputs = [residuals_flat]
        self.objective.compute_jacobian_autodiff(tape, self.model, jacobian_autodiff, 0)
        
        # Compute Jacobian analytically
        self.objective.compute_jacobian_analytic(self.state, self.model, jacobian_analytic, 0)
        
        # Compare
        J_auto = jacobian_autodiff.numpy()
        J_analytic = jacobian_analytic.numpy()
        
        print("\n=== Jacobian Comparison ===")
        print(f"Joint velocities shape: {len(self.model.joint_qd)}")
        print(f"Joint positions: {self.model.joint_q.numpy()}")
        
        # Get actual EE position
        body_tf = self.state.body_q.numpy()[self.ee_link_index]
        ee_pos = wp.transform_point(
            wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
            self.ee_offset
        )
        print(f"\nCurrent EE position: {ee_pos}")
        print(f"Target position: {self.target_position}")
        print(f"Residuals: {residuals.numpy()[0]}")
        
        print(f"\nAutodiff Jacobian shape: {J_auto.shape}")
        print("First few columns (base motion):")
        print(J_auto[0, :, :7])
        print("\nRevolute joint columns:")
        print(J_auto[0, :, 7:])
        
        print(f"\nAnalytic Jacobian shape: {J_analytic.shape}")
        print("First few columns (base motion):")
        print(J_analytic[0, :, :7])
        print("\nRevolute joint columns:")
        print(J_analytic[0, :, 7:])
        
        # Check that they match
        np.testing.assert_allclose(
            J_analytic[0], J_auto[0], 
            rtol=4e-3, atol=1e-6,
            err_msg="Jacobians don't match for floating base robot!"
        )
        
        print("\n✓ Jacobians match!")


if __name__ == "__main__":
    wp.init()
    unittest.main()