import numpy as np

class Helper():
    
    def skew_symmetric_matrix(r):
        return np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0]. 0]
        ])
    
    # body inertia tensor to global 식 (15)
    def yaw_rotation_matrix(Rz):
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])