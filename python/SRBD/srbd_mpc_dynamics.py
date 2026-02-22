"""
Single Rigid Body Dynamics (SRBD) for MPC
MIT Cheetah 3 논문 기반 구현

State: x = [θ, p, ω, v]^T (12x1)
  - θ: Body orientation (roll, pitch, yaw) [3]
  - p: Body position (x, y, z) [3]
  - ω: Angular velocity [3]
  - v: Linear velocity [3]

Control: u = [f1, f2, ..., fn]^T (3n x 1)
  - fi: Ground reaction force at foot i [3]

Dynamics: ẋ = A*x + B*u + g
"""

import numpy as np


def get_skew_symmetric_matrix(r):
    """
    벡터 r을 3x3 Skew-symmetric 행렬로 변환
    [r]_x * v = r × v (cross product)
    
    Args:
        r: 3D vector [x, y, z]
    
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])


def get_rotation_matrix_z(yaw):
    """
    Yaw 각도에 대한 회전 행렬 (Z축 회전)
    
    Args:
        yaw: Yaw angle (radians)
    
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


class SRBDModel:
    """
    Single Rigid Body Dynamics Model for MPC
    
    Continuous dynamics: ẋ = A_c*x + B_c*u + g_c
    Discrete dynamics: x_{k+1} = A*x_k + B*u_k + g
    """
    
    def __init__(self, mass, inertia, dt, n_feet=2):
        """
        Initialize SRBD model
        
        Args:
            mass: Robot mass (kg)
            inertia: Body inertia tensor (3x3 matrix)
            dt: Discretization timestep (s)
            n_feet: Number of feet (default: 2 for biped)
        """
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)
        self.dt = dt
        self.n_feet = n_feet
        
        # Gravity
        self.g_accel = 9.81  # m/s^2
        
        # State dimension: 12
        # Control dimension: 3 * n_feet
        self.nx = 12
        self.nu = 3 * n_feet
    
    def compute_A_matrix(self, yaw):
        """
        Compute A matrix (state transition matrix)
        
        Continuous:
        A_c = [0   0   Rz(ψ)  0  ]
              [0   0   0       I  ]
              [0   0   0       0  ]
              [0   0   0       0  ]
        
        Discrete: A = I + A_c * dt
        
        Args:
            yaw: Current yaw angle (radians)
        
        Returns:
            A: 12x12 state transition matrix
        """
        A = np.eye(12)
        
        # Rotation matrix (yaw only)
        Rz = get_rotation_matrix_z(yaw)
        
        # dθ/dt = Rz(ψ) * ω
        A[0:3, 6:9] = Rz * self.dt
        
        # dp/dt = v
        A[3:6, 9:12] = np.eye(3) * self.dt
        
        return A
    
    def compute_B_matrix(self, foot_positions):
        """
        Compute B matrix (control input matrix)
        
        Continuous:
        B_c = [0                                    ]
              [0                                    ]
              [I^-1*[r1]_x  I^-1*[r2]_x  ...       ]
              [I/m          I/m          ...       ]
        
        Discrete: B = B_c * dt
        
        Args:
            foot_positions: List of foot positions relative to CoM
                           [(x1,y1,z1), (x2,y2,z2), ...]
                           Each is 3D vector in body frame
        
        Returns:
            B: 12x(3*n_feet) control input matrix
        """
        B = np.zeros((12, self.nu))
        
        for i in range(self.n_feet):
            r_foot = foot_positions[i]
            
            # Angular velocity: dω/dt = I^-1 * (r × f)
            # [r]_x * f = r × f
            r_skew = get_skew_symmetric_matrix(r_foot)
            B[6:9, i*3:(i+1)*3] = self.inertia_inv @ r_skew * self.dt
            
            # Linear velocity: dv/dt = f / m
            B[9:12, i*3:(i+1)*3] = (np.eye(3) / self.mass) * self.dt
        
        return B
    
    def compute_g_vector(self):
        """
        Compute g vector (gravity and constant terms)
        
        g = [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, -g*dt]^T
        
        Returns:
            g: 12x1 gravity vector
        """
        g = np.zeros(12)
        g[11] = -self.g_accel * self.dt  # z-axis velocity affected by gravity
        
        return g
    
    def predict_next_state(self, x_current, u_control, foot_positions):
        """
        Predict next state using discrete dynamics
        
        x_{k+1} = A*x_k + B*u_k + g
        
        Args:
            x_current: Current state (12,)
            u_control: Control input (3*n_feet,)
            foot_positions: Foot positions relative to CoM
        
        Returns:
            x_next: Next state (12,)
        """
        # Extract yaw from current state
        yaw = x_current[2]  # θ[2] = yaw
        
        # Compute matrices
        A = self.compute_A_matrix(yaw)
        B = self.compute_B_matrix(foot_positions)
        g = self.compute_g_vector()
        
        # Predict next state
        x_next = A @ x_current + B @ u_control + g
        
        return x_next
    
    def get_state_from_mujoco(self, data, body_id):
        """
        Extract SRBD state from MuJoCo data
        
        Args:
            data: MuJoCo data
            body_id: Body ID for the trunk
        
        Returns:
            x: State vector (12,)
        """
        x = np.zeros(12)
        
        # Orientation (roll, pitch, yaw) from quaternion
        quat = data.xquat[body_id]  # [w, x, y, z]
        # Convert quaternion to Euler angles (simplified)
        # For full implementation, use proper conversion
        # Here we approximate with small angles
        x[0:3] = self._quat_to_euler(quat)
        
        # Position
        x[3:6] = data.xpos[body_id]
        
        # Angular velocity (body frame)
        x[6:9] = data.cvel[body_id, 3:6]
        
        # Linear velocity (world frame)
        x[9:12] = data.cvel[body_id, 0:3]
        
        return x
    
    def _quat_to_euler(self, quat):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        Simplified version - use proper library for production
        
        Args:
            quat: Quaternion [w, x, y, z]
        
        Returns:
            euler: [roll, pitch, yaw]
        """
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SRBD Model Test")
    print("=" * 70)
    
    # Robot parameters (G1 humanoid)
    mass = 12.0  # kg
    inertia = np.diag([0.5, 0.5, 0.5])  # kg*m^2 (simplified)
    dt = 0.025  # 25ms (40Hz MPC update)
    n_feet = 2  # Biped
    
    # Create model
    srbd = SRBDModel(mass, inertia, dt, n_feet)
    
    print(f"\nModel Parameters:")
    print(f"  Mass: {mass} kg")
    print(f"  Inertia: {np.diag(inertia)} kg*m^2")
    print(f"  dt: {dt} s ({1/dt:.1f} Hz)")
    print(f"  Number of feet: {n_feet}")
    print(f"  State dimension: {srbd.nx}")
    print(f"  Control dimension: {srbd.nu}")
    
    # Test state
    x_current = np.zeros(12)
    x_current[2] = 0.1  # yaw = 0.1 rad
    x_current[5] = 0.3  # z = 0.3 m (standing height)
    
    # Test control (ground reaction forces)
    u_control = np.zeros(6)
    u_control[2] = mass * 9.81 / 2  # Left foot z-force
    u_control[5] = mass * 9.81 / 2  # Right foot z-force
    
    # Foot positions (relative to CoM)
    foot_positions = [
        np.array([0.0, 0.1, -0.3]),   # Left foot
        np.array([0.0, -0.1, -0.3])   # Right foot
    ]
    
    # Compute matrices
    print("\n" + "=" * 70)
    print("Computing Dynamics Matrices")
    print("=" * 70)
    
    A = srbd.compute_A_matrix(x_current[2])
    B = srbd.compute_B_matrix(foot_positions)
    g = srbd.compute_g_vector()
    
    print(f"\nA matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")
    print(f"g vector shape: {g.shape}")
    
    print(f"\nA matrix (non-zero elements):")
    print(f"  A[0:3, 6:9] (Rz*dt):\n{A[0:3, 6:9]}")
    print(f"  A[3:6, 9:12] (I*dt):\n{A[3:6, 9:12]}")
    
    print(f"\nB matrix (angular velocity part):")
    print(f"  B[6:9, 0:3] (left foot):\n{B[6:9, 0:3]}")
    print(f"  B[6:9, 3:6] (right foot):\n{B[6:9, 3:6]}")
    
    print(f"\nB matrix (linear velocity part):")
    print(f"  B[9:12, 0:3] (left foot):\n{B[9:12, 0:3]}")
    print(f"  B[9:12, 3:6] (right foot):\n{B[9:12, 3:6]}")
    
    print(f"\ng vector:")
    print(f"  {g}")
    
    # Predict next state
    x_next = srbd.predict_next_state(x_current, u_control, foot_positions)
    
    print("\n" + "=" * 70)
    print("State Prediction")
    print("=" * 70)
    print(f"\nCurrent state:")
    print(f"  θ (roll, pitch, yaw): {x_current[0:3]}")
    print(f"  p (x, y, z): {x_current[3:6]}")
    print(f"  ω (angular vel): {x_current[6:9]}")
    print(f"  v (linear vel): {x_current[9:12]}")
    
    print(f"\nNext state:")
    print(f"  θ (roll, pitch, yaw): {x_next[0:3]}")
    print(f"  p (x, y, z): {x_next[3:6]}")
    print(f"  ω (angular vel): {x_next[6:9]}")
    print(f"  v (linear vel): {x_next[9:12]}")
    
    print("\n" + "=" * 70)
