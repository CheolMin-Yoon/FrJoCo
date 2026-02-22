"""
Layer 3: Dynamics-based Whole-Body Control (Torque QP WBC)
Task Space Impedance Control을 QP로 구현

Based on:
- Indy7 task_space_impedance.py
- Operational Space Control
- Hierarchical QP formulation

Control Law:
  τ = J^T * F_task + N^T * τ_null + τ_gravity
  
where:
  F_task: Task space forces (from impedance control)
  N: Nullspace projection matrix
  τ_null: Nullspace torques (posture control)
"""

import numpy as np
import mujoco

# Try to import qpsolvers, but provide fallback
try:
    import qpsolvers
    HAS_QPSOLVERS = True
except ImportError:
    HAS_QPSOLVERS = False
    print("Warning: qpsolvers not found. Using direct torque computation without QP optimization.")


class DynamicsWBC:
    """
    Dynamics-based Whole-Body Controller using QP
    
    Solves:
      min  ||M^(1/2) * (τ - τ_des)||^2
      s.t. τ_min ≤ τ ≤ τ_max
           Contact constraints (if needed)
    """
    
    def __init__(
        self,
        model,
        data,
        # Task impedance gains (reduced for stability)
        com_stiffness=50.0,
        com_damping=14.1,
        foot_stiffness=100.0,
        foot_damping=20.0,
        orientation_stiffness=30.0,
        orientation_damping=10.0,
        # Nullspace gains (reduced for stability)
        posture_stiffness=5.0,
        posture_damping=3.2,
        # QP weights
        task_weight=1.0,
        nullspace_weight=0.01,
        regularization=1e-6,
    ):
        self.model = model
        self.data = data
        
        # Task gains
        self.Kp_com = com_stiffness * np.eye(3)
        self.Kd_com = com_damping * np.eye(3)
        
        self.Kp_foot = foot_stiffness * np.eye(3)
        self.Kd_foot = foot_damping * np.eye(3)
        
        self.Kp_ori = orientation_stiffness * np.eye(3)
        self.Kd_ori = orientation_damping * np.eye(3)
        
        # Nullspace gains
        self.Kp_null = posture_stiffness * np.eye(model.nv)
        self.Kd_null = posture_damping * np.eye(model.nv)
        
        # QP weights
        self.w_task = task_weight
        self.w_null = nullspace_weight
        self.reg = regularization
        
        # Try to get site IDs, fallback to body IDs
        self.use_bodies = False
        try:
            self.left_foot_site = model.site("left_foot").id
            self.right_foot_site = model.site("right_foot").id
        except:
            # Fallback to using body positions
            self.use_bodies = True
            self.left_foot_body = model.body("left_ankle_roll_link").id
            self.right_foot_body = model.body("right_ankle_roll_link").id
        
        # Desired posture (from keyframe)
        self.q_des = None
        
        # Memory allocation
        self.M_inv = np.zeros((model.nv, model.nv))
        
    def initialize_from_keyframe(self, keyframe_name="knees_bent"):
        """Initialize desired posture from keyframe"""
        key_id = self.model.key(keyframe_name).id
        self.q_des = self.model.key(keyframe_name).qpos.copy()
        
        # Reset data to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
    
    def compute_com_jacobian(self):
        """
        Compute CoM Jacobian
        
        Returns:
            J_com: (3 x nv) CoM Jacobian
        """
        J_com = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, J_com, 0)  # Root body
        return J_com
    
    def compute_foot_jacobian(self, site_or_body_id, is_body=False):
        """
        Compute foot Jacobian (position only)
        
        Args:
            site_or_body_id: Site ID or Body ID for foot
            is_body: If True, use body Jacobian instead of site Jacobian
        
        Returns:
            J_foot: (3 x nv) foot Jacobian
        """
        J_foot = np.zeros((3, self.model.nv))
        if is_body:
            mujoco.mj_jacBody(self.model, self.data, J_foot, None, site_or_body_id)
        else:
            mujoco.mj_jacSite(self.model, self.data, J_foot, None, site_or_body_id)
        return J_foot
    
    def compute_task_space_inertia(self, J):
        """
        Compute task space inertia matrix
        
        Λ = (J * M^-1 * J^T)^-1
        
        Args:
            J: Jacobian matrix (m x nv)
        
        Returns:
            Lambda: Task space inertia (m x m)
        """
        # Compute M^-1
        mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        
        # Λ^-1 = J * M^-1 * J^T
        Lambda_inv = J @ self.M_inv @ J.T
        
        # Invert with singularity handling
        if abs(np.linalg.det(Lambda_inv)) >= 1e-4:
            Lambda = np.linalg.inv(Lambda_inv)
        else:
            Lambda = np.linalg.pinv(Lambda_inv, rcond=1e-4)
        
        return Lambda
    
    def compute_nullspace_projection(self, J, Lambda):
        """
        Compute dynamically consistent nullspace projection
        
        N = I - J^T * Jbar^T
        where Jbar = M^-1 * J^T * Lambda
        
        Args:
            J: Jacobian matrix (m x nv)
            Lambda: Task space inertia (m x m)
        
        Returns:
            N: Nullspace projection matrix (nv x nv)
        """
        # Dynamically consistent pseudoinverse
        Jbar = self.M_inv @ J.T @ Lambda
        
        # Nullspace projection
        N = np.eye(self.model.nv) - J.T @ Jbar.T
        
        return N
    
    def compute_task_torque(
        self,
        target_com,
        target_left_foot,
        target_right_foot,
    ):
        """
        Compute task space control torques
        
        τ_task = Σ J_i^T * F_i
        
        where F_i = Λ_i * (Kp * e_i + Kd * ė_i)
        
        Args:
            target_com: Target CoM position (3,)
            target_left_foot: Target left foot position (3,)
            target_right_foot: Target right foot position (3,)
        
        Returns:
            tau_task: Task torques (nv,)
            J_all: Stacked Jacobian for all tasks
            Lambda_all: Block diagonal task space inertia
        """
        tau_task = np.zeros(self.model.nv)
        
        # List to store Jacobians
        J_list = []
        Lambda_list = []
        
        # ================================================================
        # CoM Task
        # ================================================================
        J_com = self.compute_com_jacobian()
        
        # CoM error
        com_current = self.data.subtree_com[0]
        e_com = target_com - com_current
        
        # CoM velocity error
        v_com = J_com @ self.data.qvel
        ed_com = -v_com  # Target velocity = 0
        
        # Task space inertia
        Lambda_com = self.compute_task_space_inertia(J_com)
        
        # Task force
        F_com = Lambda_com @ (self.Kp_com @ e_com + self.Kd_com @ ed_com)
        
        # Task torque
        tau_task += J_com.T @ F_com
        
        J_list.append(J_com)
        Lambda_list.append(Lambda_com)
        
        # ================================================================
        # Left Foot Task
        # ================================================================
        if self.use_bodies:
            J_left = self.compute_foot_jacobian(self.left_foot_body, is_body=True)
            p_left = self.data.body(self.left_foot_body).xpos
        else:
            J_left = self.compute_foot_jacobian(self.left_foot_site, is_body=False)
            p_left = self.data.site(self.left_foot_site).xpos
        
        # Foot error
        e_left = target_left_foot - p_left
        
        # Foot velocity error
        v_left = J_left @ self.data.qvel
        ed_left = -v_left
        
        # Task space inertia
        Lambda_left = self.compute_task_space_inertia(J_left)
        
        # Task force
        F_left = Lambda_left @ (self.Kp_foot @ e_left + self.Kd_foot @ ed_left)
        
        # Task torque
        tau_task += J_left.T @ F_left
        
        J_list.append(J_left)
        Lambda_list.append(Lambda_left)
        
        # ================================================================
        # Right Foot Task
        # ================================================================
        if self.use_bodies:
            J_right = self.compute_foot_jacobian(self.right_foot_body, is_body=True)
            p_right = self.data.body(self.right_foot_body).xpos
        else:
            J_right = self.compute_foot_jacobian(self.right_foot_site, is_body=False)
            p_right = self.data.site(self.right_foot_site).xpos
        
        # Foot error
        e_right = target_right_foot - p_right
        
        # Foot velocity error
        v_right = J_right @ self.data.qvel
        ed_right = -v_right
        
        # Task space inertia
        Lambda_right = self.compute_task_space_inertia(J_right)
        
        # Task force
        F_right = Lambda_right @ (self.Kp_foot @ e_right + self.Kd_foot @ ed_right)
        
        # Task torque
        tau_task += J_right.T @ F_right
        
        J_list.append(J_right)
        Lambda_list.append(Lambda_right)
        
        # Stack Jacobians
        J_all = np.vstack(J_list)
        Lambda_all = np.block([
            [Lambda_com, np.zeros((3, 6))],
            [np.zeros((3, 3)), Lambda_left, np.zeros((3, 3))],
            [np.zeros((3, 6)), Lambda_right]
        ])
        
        return tau_task, J_all, Lambda_all
    
    def compute_nullspace_torque(self, N):
        """
        Compute nullspace torques for posture control
        
        τ_null = N^T * (Kp * (q_des - q) - Kd * qvel)
        
        Args:
            N: Nullspace projection matrix (nv x nv)
        
        Returns:
            tau_null: Nullspace torques (nv,)
        """
        if self.q_des is None:
            return np.zeros(self.model.nv)
        
        # Posture error in velocity space
        # For floating base, we need to handle quaternion difference properly
        q_error_vel = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, q_error_vel, 1.0, self.data.qpos, self.q_des)
        
        # Posture control in velocity space
        tau_posture = self.Kp_null @ q_error_vel - self.Kd_null @ self.data.qvel
        
        # Project to nullspace
        tau_null = N.T @ tau_posture
        
        return tau_null
    
    def solve_qp(
        self,
        target_com,
        target_left_foot,
        target_right_foot,
    ):
        """
        Solve QP to compute optimal torques (or use direct computation if QP not available)
        
        min  ||M^(1/2) * (τ - τ_des)||^2 + w_reg * ||τ||^2
        s.t. τ_min ≤ τ ≤ τ_max
        
        Args:
            target_com: Target CoM position (3,)
            target_left_foot: Target left foot position (3,)
            target_right_foot: Target right foot position (3,)
        
        Returns:
            tau: Optimal torques (nv,)
        """
        # Compute task torques
        tau_task, J_all, Lambda_all = self.compute_task_torque(
            target_com, target_left_foot, target_right_foot
        )
        
        # Compute nullspace projection
        Lambda_combined = self.compute_task_space_inertia(J_all)
        N = self.compute_nullspace_projection(J_all, Lambda_combined)
        
        # Compute nullspace torques
        tau_null = self.compute_nullspace_torque(N)
        
        # Gravity compensation
        tau_gravity = self.data.qfrc_bias.copy()
        
        # Desired torque
        tau_des = tau_task + tau_null + tau_gravity
        
        # Bounds
        lb = self.model.actuator_ctrlrange[:, 0]
        ub = self.model.actuator_ctrlrange[:, 1]
        
        if HAS_QPSOLVERS:
            # Use QP solver if available
            # Get mass matrix
            M = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M, self.data.qM)
            
            # QP formulation:
            # min  (1/2) * τ^T * P * τ - q^T * τ
            # s.t. lb ≤ τ ≤ ub
            
            # P = M + reg * I (mass-weighted + regularization)
            P = M + self.reg * np.eye(self.model.nv)
            
            # q = M * τ_des
            q = M @ tau_des
            
            # Solve QP
            try:
                tau = qpsolvers.solve_qp(
                    P, -q, None, None, None, None, lb, ub,
                    solver="daqp"
                )
                if tau is None:
                    print("QP failed, using desired torque")
                    tau = tau_des
            except:
                print("QP solver error, using desired torque")
                tau = tau_des
        else:
            # Direct computation without QP
            tau = tau_des
        
        # Extract actuated joints only (skip floating base: first 6 DOF)
        # tau is (nv,) = 35, but we need (nu,) = 29 for actuators
        tau_actuated = tau[6:]  # Skip floating base velocities
        
        # Clip to actuator limits
        tau_actuated = np.clip(tau_actuated, lb, ub)
        
        return tau_actuated


# Example usage
if __name__ == "__main__":
    print("Dynamics-based WBC requires MuJoCo model to run")
    print("Use this class in your main control loop")
