import numpy as np
import warp as wp
import unittest
from ik_objectives import JointLimitObjective


class FakeModel:
    """Minimal model for testing joint limit objective"""
    def __init__(self, num_envs, dof_per_env):
        self.num_envs = num_envs
        self.dof_per_env = dof_per_env
        total_joints = num_envs * dof_per_env
        
        # Create some test joint positions - some within limits, some at limits
        joint_q_np = np.array([
            0.5,   # Within limits
            1.0,   # At upper limit
            -1.0,  # At lower limit
            0.0,   # Within limits
            0.9,   # Close to upper
            -0.9,  # Close to lower
        ] * num_envs)[:total_joints]
        
        self.joint_q = wp.array(joint_q_np, dtype=wp.float32, requires_grad=True)
        self.joint_qd = wp.zeros(total_joints, dtype=wp.float32)
        
        # Set joint limits
        lower_limits = np.full(dof_per_env, -1.0, dtype=np.float32)
        upper_limits = np.full(dof_per_env, 1.0, dtype=np.float32)
        
        self.joint_limit_lower = wp.array(lower_limits, dtype=wp.float32)
        self.joint_limit_upper = wp.array(upper_limits, dtype=wp.float32)


class FakeState:
    """Minimal state for testing"""
    def __init__(self):
        pass  # Joint limit objective doesn't use state


class TestJointLimitJacobian(unittest.TestCase):
    def setUp(self):
        self.num_envs = 2
        self.dof_per_env = 6
        self.total_residuals = self.dof_per_env  # Joint limits only
        
        self.model = FakeModel(self.num_envs, self.dof_per_env)
        self.state = FakeState()
        
        # Create objective
        self.objective = JointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            dof_per_env=self.dof_per_env,
            weight=0.1,
            num_envs=self.num_envs,
            total_residuals=self.total_residuals,
            residual_offset=0
        )
    
    def test_jacobian_comparison(self):
        """Compare autodiff and analytic Jacobians"""
        # Allocate arrays
        residuals = wp.zeros((self.num_envs, self.total_residuals), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((self.num_envs, self.total_residuals, self.dof_per_env), dtype=wp.float32)
        jacobian_analytic = wp.zeros((self.num_envs, self.total_residuals, self.dof_per_env), dtype=wp.float32)
        
        # Compute residuals
        self.objective.compute_residuals(self.state, self.model, residuals, 0, self.num_envs)
        
        # Compute Jacobian with autodiff
        tape = wp.Tape()
        with tape:
            residuals_compute = wp.zeros((self.num_envs, self.total_residuals), dtype=wp.float32, requires_grad=True)
            self.objective.compute_residuals(self.state, self.model, residuals_compute, 0, self.num_envs)
            residuals_flat = residuals_compute.flatten()
        
        tape.outputs = [residuals_flat]
        self.objective.compute_jacobian_autodiff(tape, self.model, jacobian_autodiff, 0, 
                                                self.dof_per_env, self.num_envs)
        
        # Compute Jacobian analytically
        self.objective.compute_jacobian_analytic(self.state, self.model, jacobian_analytic, 0,
                                               self.dof_per_env, self.num_envs)
        
        # Compare
        J_auto = jacobian_autodiff.numpy()
        J_analytic = jacobian_analytic.numpy()
        
        # Print for debugging
        print("\nAutodiff Jacobian:")
        print(J_auto)
        print("\nAnalytic Jacobian:")
        print(J_analytic)
        
        # Check if they match
        np.testing.assert_allclose(J_analytic, J_auto, rtol=1e-5, atol=1e-7,
                                 err_msg="Analytic and autodiff Jacobians don't match!")


if __name__ == "__main__":
    wp.init()
    unittest.main()