import numpy as np
import warp as wp
import unittest
import newton
import newton.core.articulation
from ik_objectives import PositionObjective, JointLimitObjective
from ik_solver import create_ik_solver, JacobianMode


class TestMultipleTargetsJacobian(unittest.TestCase):
    def setUp(self):
        self.num_envs = 1
        
        # Build a 4-link robot with 2 end-effectors (branching structure)
        builder = newton.ModelBuilder()
        
        # Base (fixed)
        base = builder.add_body(
            xform=wp.transform_identity(),
            mass=0.0,  # Fixed
            key="base"
        )
        
        # Link 1
        link1 = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()),
            mass=1.0,
            key="link1"
        )
        joint1 = builder.add_joint_revolute(
            parent=-1,
            child=link1,
            parent_xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 0.0, 1.0],
            key="joint1"
        )
        
        # Branch A - Link 2a (first end-effector)
        link2a = builder.add_body(
            xform=wp.transform([0.5, 0.0, 1.0], wp.quat_identity()),
            mass=0.5,
            key="link2a"
        )
        joint2a = builder.add_joint_revolute(
            parent=link1,
            child=link2a,
            parent_xform=wp.transform([0.5, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 1.0, 0.0],
            key="joint2a"
        )
        
        # Branch B - Link 2b (second end-effector)  
        link2b = builder.add_body(
            xform=wp.transform([-0.5, 0.0, 1.0], wp.quat_identity()),
            mass=0.5,
            key="link2b"
        )
        joint2b = builder.add_joint_revolute(
            parent=link1,
            child=link2b,
            parent_xform=wp.transform([-0.5, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 1.0, 0.0],
            key="joint2b"
        )
        
        self.model = builder.finalize(requires_grad=True)
        print(f"Joint ancestors: {self.model.joint_ancestor.numpy()}")
        print(f"Joint qd_start: {self.model.joint_qd_start.numpy()}")
        print(f"joint_child: {self.model.joint_child.numpy()}")
        print(f"joint_parent: {self.model.joint_parent.numpy()}")


        
        # Set initial joint positions
        self.model.joint_q = wp.array([0.0, 0.3, -0.3], dtype=wp.float32, requires_grad=True)
        self.model.joint_qd = wp.zeros(3, dtype=wp.float32)
        
        # Compute FK
        self.state = self.model.state()
        newton.core.articulation.eval_fk(
            self.model, 
            self.model.joint_q, 
            self.model.joint_qd, 
            self.state, 
            None
        )
        
        # Setup multiple end-effectors
        self.num_links = 4  # link1, link2a, link2b
        self.ee_info = [
            {"link_index": 2, "offset": wp.vec3(0.5, 0.0, 0.0)},  # Tip of link2a
            {"link_index": 3, "offset": wp.vec3(-0.5, 0.0, 0.0)}, # Tip of link2b
        ]
        self.num_ees = len(self.ee_info)
        self.coords = 3
        
        # Get initial positions and set targets
        self.targets = []
        for i, ee in enumerate(self.ee_info):
            body_idx = ee["link_index"]
            body_tf = self.state.body_q.numpy()[body_idx]
            ee_pos = wp.transform_point(
                wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                ee["offset"]
            )
            # Set different targets for each EE
            target = ee_pos + wp.vec3(0.1, 0.05, -0.1) * float(i + 1)
            self.targets.append(target)
        
        # Create objectives
        self.objectives = []
        total_residuals = self.num_ees * 3 + self.coords  # Position residuals
        
        # Add position objectives
        for i, ee in enumerate(self.ee_info):
            obj = PositionObjective(
                link_index=ee["link_index"],
                link_offset=ee["offset"],
                target_positions=wp.array([self.targets[i]], dtype=wp.vec3),  # Single env
                num_links=self.num_links,
                num_envs=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=i * 3
            )
            self.objectives.append(obj)
        
        # Add joint limit objective
        joint_limit_obj = JointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=0.1,
            num_envs=self.num_envs,
            total_residuals=total_residuals,
            residual_offset=self.num_ees * 3
        )
        self.objectives.append(joint_limit_obj)
    
    def test_jacobian_comparison(self):
        """Compare autodiff and analytic Jacobians for multiple targets"""
        print("\n=== Testing Multiple Target Jacobians ===")
        print(f"Robot has {self.num_ees} end-effectors, {self.coords} coords")
        
        # Create solver with autodiff
        solver_autodiff = create_ik_solver(
            model=self.model,
            num_envs=self.num_envs,
            objectives=self.objectives,
            damping=1e-3,
            jacobian_mode=JacobianMode.AUTODIFF
        )
        
        # Create solver with analytic
        solver_analytic = create_ik_solver(
            model=self.model,
            num_envs=self.num_envs,
            objectives=self.objectives,
            damping=1e-3,
            jacobian_mode=JacobianMode.ANALYTIC
        )
        
        # Compute residuals (should be same for both)
        residuals_auto = solver_autodiff.compute_residuals()
        residuals_anal = solver_analytic.compute_residuals()
        
        print("\nResiduals:")
        print(residuals_auto.numpy())
        
        # Compute Jacobians
        jacobian_auto = solver_autodiff.compute_jacobian()
        jacobian_anal = solver_analytic.compute_jacobian()
        
        # Compare
        J_auto = jacobian_auto.numpy()
        J_anal = jacobian_anal.numpy()
        
        print(f"\nJacobian shape: {J_auto.shape}")
        print("\nAutodiff Jacobian:")
        print(J_auto[0])
        print("\nAnalytic Jacobian:")
        print(J_anal[0])
        
        # Check each objective's contribution
        for i in range(self.num_ees):
            print(f"\nEE {i} Jacobian rows (positions {i*3} to {i*3+2}):")
            print("Autodiff:", J_auto[0, i*3:(i+1)*3, :])
            print("Analytic:", J_anal[0, i*3:(i+1)*3, :])
        
        # Test that they match
        np.testing.assert_allclose(
            J_anal, J_auto,
            rtol=1e-4, atol=1e-6,
            err_msg="Analytic and autodiff Jacobians don't match for multiple targets!"
        )
        
        print("\n✓ Multiple target Jacobians match!")


if __name__ == "__main__":
    wp.init()
    unittest.main()