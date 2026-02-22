import numpy as np
import mujoco as mj
import qpsolver as 


    # 메모리 할당
    jac = np.zeros((6, model.nv))           #
    twist_error = np.zeros(6)                     # 
    site_quat = np.zeros(4)                 #
    site_quat_conj = np.zeros(4)            #
    error_quat = np.zeros(4)                #
    M_inv = np.zeros((model.nv, model.nv))  #
    Λ = np.zeros((6, 6)) 

def skew_symmetric_matrix(r):
    return np.array([
        
    ])


class SingleRigidBodyModel(self):
    
    # 메모리 할당
    
    A = np.zeros((12, 12))
    B = np.zeros(())
    
    A = [theta, p, Omega, dp]
    
    A = [qpos[6:], qpos[3:], 몰라, qvel[:3]] # 12x12
    
    Rz = np.array([np.cos(psi), -np.])
    
    dx = [theta, p, Omega, dp]
    
    A = [0 0 0 0 0 0 Rz(yaw) 0 0 0
         0 0 0 0 0 0 0 0 0 1 1 1
         0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0]
    
    B = [0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0
         inertia^-1[r1]_x ... inertia^-1[rn]_x
         1/m 1/m?....]
    
    g = [0 0 0 9.81]