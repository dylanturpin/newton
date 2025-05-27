import numpy as np
import warp as wp
import unittest
import newton
import newton.core.articulation
import newton.utils
import newton.examples
from ik_objectives import PositionObjective
import os


class TestPositionJacobianMJCF(unittest.TestCase):
    def setUp(self):
        self.num_envs = 3
        
        # Build model from MJCF
        self.model, self.num_links, self.ee_link_index, self.ee_offset, self.coords = self._build_model(self.num_envs)
        
        # Compute forward kinematics
        self.state = self.model.state()

        print(f"Joint child: {self.model.joint_child.numpy()}")
        print(f"Joint parent: {self.model.joint_parent.numpy()}")
        print(f"Body count: {len(self.state.body_q)}")

        newton.core.articulation.eval_fk(
            self.model, 
            self.model.joint_q, 
            self.model.joint_qd, 
            self.state, 
            None
        )
        
        # Get initial EE positions and set targets
        self.target_positions = self._initialize_targets(self.num_envs)
        
        # Create position objective
        self.total_residuals = 3  # Position only (x, y, z)
        self.objective = PositionObjective(
            link_index=self.ee_link_index,
            link_offset=self.ee_offset,
            target_positions=self.target_positions,
            num_links=self.num_links,
            num_envs=self.num_envs,
            total_residuals=self.total_residuals,
            residual_offset=0
        )
        
        # Override supports_analytic to return True
        self.objective.supports_analytic = lambda: True
    
    def _build_model(self, num_envs):
        """Build robot model from MJCF"""
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1
        
        # Load simple robot from MJCF
        mjcf_path = os.path.join(os.path.dirname(__file__), "simple_robot.xml")
        newton.utils.parse_mjcf(
            mjcf_path,
            articulation_builder,
            floating=True,  # Fixed base
            armature_scale=1.0,
            scale=1.0,
        )
        
        # Set initial joint positions
        initial_joint_positions = [
            0.0,   # joint1 (revolute)
            0.2,   # joint2 (prismatic)
            0.3,   # joint3 (revolute)
        ]
        
        for i, value in enumerate(initial_joint_positions):
            articulation_builder.joint_q[i] = value
        
        builder = newton.ModelBuilder()
        
        # Get robot properties
        num_links = len(articulation_builder.body_q)
        coords = len(articulation_builder.joint_q)
        
        # End-effector is the last link (link2)
        ee_link_index = num_links - 1  # Last body
        ee_offset = wp.vec3(0.0, 0.0, 0.2)  # Tip of link2
        
        # Position environments
        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(2.0, 0.0, 0.0))
        
        # Add articulations with different initial configurations
        for i in range(num_envs):
            # Modify initial positions slightly for each environment
            for j in range(coords):
                articulation_builder.joint_q[j] = initial_joint_positions[j] + i * 0.1
            
            builder.add_builder(
                articulation_builder, 
                xform=wp.transform(positions[i], wp.quat_identity())
            )
        
        model = builder.finalize(requires_grad=True)
        model.ground = True
        
        return model, num_links, ee_link_index, ee_offset, coords
    
    def _initialize_targets(self, num_envs):
        """Initialize targets from current end-effector positions"""
        target_positions = []
        
        for env_idx in range(num_envs):
            # Get actual EE position
            body_idx = env_idx * self.num_links + self.ee_link_index
            body_tf = self.state.body_q.numpy()[body_idx]
            ee_pos = wp.transform_point(
                wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                self.ee_offset
            )
            
            # Set target slightly offset from current position
            target = ee_pos + wp.vec3(0.1, 0.05, -0.05)
            target_positions.append(target)
        
        return wp.array(target_positions, dtype=wp.vec3)
    
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
        
        # Compare for each environment
        J_auto = jacobian_autodiff.numpy()
        J_analytic = jacobian_analytic.numpy()
        
        print("\nModel info:")
        print(f"Number of links per env: {self.num_links}")
        print(f"Coords per env: {self.coords // self.num_envs}")
        print(f"Joint types: {self.model.joint_type.numpy()[:self.coords]}")
        print(f"Joint names: joint1(revolute), joint2(prismatic), joint3(revolute)")
        
        for env_idx in range(self.num_envs):
            print(f"\n=== Environment {env_idx} ===")
            print(f"Joint positions: {self.model.joint_q.numpy()[env_idx*self.coords:(env_idx+1)*self.coords]}")
            
            # Get actual EE position
            body_idx = env_idx * self.num_links + self.ee_link_index
            body_tf = self.state.body_q.numpy()[body_idx]
            ee_pos = wp.transform_point(
                wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                self.ee_offset
            )
            print(f"Current EE position: {ee_pos}")
            print(f"Target position: {self.target_positions.numpy()[env_idx]}")
            print(f"Residuals: {residuals.numpy()[env_idx]}")
            
            print(f"\nAutodiff Jacobian:")
            print(J_auto[env_idx])
            print(f"Analytic Jacobian:")
            print(J_analytic[env_idx])
            
            # Check this environment
            np.testing.assert_allclose(
                J_analytic[env_idx], J_auto[env_idx], 
                rtol=1e-4, atol=1e-6,
                err_msg=f"Jacobians don't match for environment {env_idx}!"
            )
        
        print("\n✓ All environments passed!")


if __name__ == "__main__":
    wp.init()
    unittest.main()