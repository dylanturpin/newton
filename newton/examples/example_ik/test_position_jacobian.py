import numpy as np
import warp as wp
import unittest
import newton
import newton.core.articulation
from ik_objectives import PositionObjective


class TestPositionJacobianMultiEnv(unittest.TestCase):
    def setUp(self):
        self.num_envs = 3
        
        # Build multiple 2-joint planar robots
        builder = newton.ModelBuilder()
        
        # Store info for each environment
        self.link_indices = []
        
        for env_idx in range(self.num_envs):
            # Offset each robot in space
            x_offset = env_idx * 3.0
            
            # Link 1 - 1m long
            link1 = builder.add_body(
                xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()),
                mass=1.0,
                key=f"link1_env{env_idx}"
            )
            
            # Joint 1 - revolute joint at origin
            joint1 = builder.add_joint_revolute(
                parent=-1,  # World frame
                child=link1,
                parent_xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint1_env{env_idx}"
            )
            
            # Add visual for link1
            builder.add_shape_box(
                body=link1,
                xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
                hx=0.5,
                hy=0.05,
                hz=0.05,
            )
            
            # Link 2 - 1m long
            link2 = builder.add_body(
                xform=wp.transform([x_offset + 1.0, 0.0, 0.0], wp.quat_identity()),
                mass=1.0,
                key=f"link2_env{env_idx}"
            )
            
            # Joint 2 - revolute joint at end of link1
            joint2 = builder.add_joint_revolute(
                parent=link1,
                child=link2,
                parent_xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint2_env{env_idx}"
            )
            
            # Add visual for link2
            builder.add_shape_box(
                body=link2,
                xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
                hx=0.5,
                hy=0.05,
                hz=0.05,
            )
            
            # Store link2 index (end-effector link)
            self.link_indices.append(link2)
        
        # Build model
        self.model = builder.finalize(requires_grad=True)

        # Add this to the test after building the model:
        print(f"Number of articulations: {len(self.model.articulation_start) - 1}")
        print(f"Articulation starts: {self.model.articulation_start.numpy()}")
        print(f"Joint ancestors: {self.model.joint_ancestor.numpy()}")
        print(f"Joint q_start: {self.model.joint_q_start.numpy()}")
        print(f"Joint qd_start: {self.model.joint_qd_start.numpy()}")

        
        # Set different initial joint angles for each environment
        joint_angles = []
        for env_idx in range(self.num_envs):
            joint_angles.extend([
                0.0 + env_idx * 0.1,      # joint1 angle
                np.pi/4 + env_idx * 0.2   # joint2 angle
            ])
        
        self.model.joint_q = wp.array(joint_angles, dtype=wp.float32, requires_grad=True)
        self.model.joint_qd = wp.zeros(len(joint_angles), dtype=wp.float32)
        
        # Compute forward kinematics
        self.state = self.model.state()
        newton.core.articulation.eval_fk(
            self.model, 
            self.model.joint_q, 
            self.model.joint_qd, 
            self.state, 
            None
        )
        
        # Setup for each environment
        self.ee_link_index = 1  # link2 within each articulation
        self.ee_offset = wp.vec3(1.0, 0.0, 0.0)  # 1m from joint to tip
        self.num_links = 2  # 2 links per articulation
        self.coords_per_env = 2  # 2 revolute joints per articulation
        self.total_residuals = 3  # Position only (x, y, z)
        
        # Create different target positions for each environment
        target_positions = []
        for env_idx in range(self.num_envs):
            x_offset = env_idx * 3.0
            target_positions.append([
                x_offset + 1.5 + env_idx * 0.1,  # x
                0.5 + env_idx * 0.1,              # y
                0.0                               # z
            ])
        
        self.target_position_array = wp.array(target_positions, dtype=wp.vec3)
        
        # Create position objective
        self.objective = PositionObjective(
            link_index=self.ee_link_index,
            link_offset=self.ee_offset,
            target_positions=self.target_position_array,
            num_links=self.num_links,
            num_envs=self.num_envs,
            total_residuals=self.total_residuals,
            residual_offset=0
        )
        
        # Override supports_analytic to return True for testing
        self.objective.supports_analytic = lambda: True
    
    def test_jacobian_comparison_multi_env(self):
        """Compare autodiff and analytic Jacobians across multiple environments"""
        # Allocate arrays
        residuals = wp.zeros((self.num_envs, self.total_residuals), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((self.num_envs, self.total_residuals, self.coords_per_env), dtype=wp.float32)
        jacobian_analytic = wp.zeros((self.num_envs, self.total_residuals, self.coords_per_env), dtype=wp.float32)
        
        # Compute residuals
        self.objective.compute_residuals(self.state, self.model, residuals, 0)
        print("\nResiduals:")
        print(residuals.numpy())
        
        # Compute Jacobian with autodiff
        tape = wp.Tape()
        with tape:
            # Need fresh state computation in tape
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
        
        # Print info for each environment
        for env_idx in range(self.num_envs):
            print(f"\n=== Environment {env_idx} ===")
            print(f"Joint angles (rad): {self.model.joint_q.numpy()[env_idx*2:(env_idx+1)*2]}")
            
            # Get actual EE position
            body_idx = env_idx * self.num_links + self.ee_link_index  # Fixed!
            body_tf = self.state.body_q.numpy()[body_idx]
            ee_pos = wp.transform_point(
                wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                self.ee_offset
            )
            print(f"Current EE position: {ee_pos}")
            print(f"Target position: {self.target_position_array.numpy()[env_idx]}")
            
            print(f"\nAutodiff Jacobian:")
            print(J_auto[env_idx])
            print(f"Analytic Jacobian:")
            print(J_analytic[env_idx])
            
            # Check this environment
            np.testing.assert_allclose(
                J_analytic[env_idx], J_auto[env_idx], 
                rtol=1e-5, atol=1e-7,
                err_msg=f"Jacobians don't match for environment {env_idx}!"
            )
        
        print("\n✓ All environments passed!")


if __name__ == "__main__":
    wp.init()
    unittest.main()