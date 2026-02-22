"""Test script to verify the setup is correct"""

import os
import sys
import numpy as np
import mujoco

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing G1 29DOF Dynamics WBC Setup")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from Layer1_DCM import TrajectoryOptimization
    print("   ✓ Layer1_DCM imported")
except Exception as e:
    print(f"   ✗ Layer1_DCM failed: {e}")
    sys.exit(1)

try:
    from Layer2_DCM import SimplifiedModelControl
    print("   ✓ Layer2_DCM imported")
except Exception as e:
    print(f"   ✗ Layer2_DCM failed: {e}")
    sys.exit(1)

try:
    from Layer3_dynamics_wbc import DynamicsWBC
    print("   ✓ Layer3_dynamics_wbc imported")
except Exception as e:
    print(f"   ✗ Layer3_dynamics_wbc failed: {e}")
    sys.exit(1)

# Test 2: Load model
print("\n2. Testing model loading...")
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
xml_path = os.path.join(
    workspace_root,
    'reference', 'unitree_mujoco', 'unitree_robots', 'g1', 'g1_29dof.xml'
)

if not os.path.exists(xml_path):
    print(f"   ✗ Model file not found: {xml_path}")
    sys.exit(1)

print(f"   Model path: {xml_path}")

# Load and modify XML to add foot sites
with open(xml_path, 'r') as f:
    xml_string = f.read()

left_foot_site = '<site name="left_foot" pos="0.05 0 -0.05" size="0.01"/>'
right_foot_site = '<site name="right_foot" pos="0.05 0 -0.05" size="0.01"/>'

xml_string = xml_string.replace(
    '<body name="left_ankle_roll_link" pos="0 0 -0.017558">',
    '<body name="left_ankle_roll_link" pos="0 0 -0.017558">\n                  ' + left_foot_site
)
xml_string = xml_string.replace(
    '<body name="right_ankle_roll_link" pos="0 0 -0.017558">',
    '<body name="right_ankle_roll_link" pos="0 0 -0.017558">\n                  ' + right_foot_site
)

try:
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    print(f"   ✓ Model loaded: {model.nq} DOFs, {model.nu} actuators")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    sys.exit(1)

# Test 3: Check sites
print("\n3. Testing foot sites...")
try:
    left_site = model.site("left_foot")
    right_site = model.site("right_foot")
    print(f"   ✓ left_foot site ID: {left_site.id}")
    print(f"   ✓ right_foot site ID: {right_site.id}")
except Exception as e:
    print(f"   ✗ Site check failed: {e}")
    sys.exit(1)

# Test 4: Initialize WBC
print("\n4. Testing WBC initialization...")
try:
    data.qpos[2] = 0.75
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    wbc = DynamicsWBC(model, data)
    wbc.q_des = data.qpos.copy()
    
    com_init = data.subtree_com[0].copy()
    left_foot_init = data.site("left_foot").xpos.copy()
    right_foot_init = data.site("right_foot").xpos.copy()
    
    print(f"   ✓ WBC initialized")
    print(f"   CoM: {com_init}")
    print(f"   Left foot: {left_foot_init}")
    print(f"   Right foot: {right_foot_init}")
except Exception as e:
    print(f"   ✗ WBC initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Compute torque
print("\n5. Testing torque computation...")
try:
    tau = wbc.solve_qp(com_init, left_foot_init, right_foot_init)
    print(f"   ✓ Torque computed: shape={tau.shape}, range=[{tau.min():.2f}, {tau.max():.2f}]")
except Exception as e:
    print(f"   ✗ Torque computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Trajectory generation
print("\n6. Testing trajectory generation...")
try:
    planner = TrajectoryOptimization(
        z_c=0.75, step_time=0.7, dsp_time=0.1,
        step_height=0.08, dt=0.002
    )
    
    footsteps, dcm_ref, dcm_vel_ref, com_ref, com_vel_ref, lf_traj, rf_traj = \
        planner.compute_all_trajectories(
            n_steps=4, step_length=0.1,
            step_width=0.1185, init_com=com_init,
            init_lf=left_foot_init, init_rf=right_foot_init
        )
    
    print(f"   ✓ Trajectory generated: {len(com_ref)} steps")
    print(f"   CoM trajectory shape: {com_ref.shape}")
    print(f"   Left foot trajectory shape: {lf_traj.shape}")
    print(f"   Right foot trajectory shape: {rf_traj.shape}")
except Exception as e:
    print(f"   ✗ Trajectory generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run: python g1_wbc_dynamics.py")
