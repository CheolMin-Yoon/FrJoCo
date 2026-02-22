import numpy as np
import mujoco
import mink

class WholeBodyController:
    """
    Layer 3: Whole-Body Control (Kinematic Level)
    Mink를 사용하여 Task Space 명령(CoM, Foot)을 Joint Space 명령(qvel)으로 변환
    """
    def __init__(
        self, 
        model, 
        data,
        # Task costs (개별 설정 가능)
        com_cost: float = 100.0,
        foot_position_cost: float = 200.0, # com 100, foot 200 일때 성공
        foot_orientation_cost: float = 5.0,
        pelvis_orientation_cost: float = 0.0,
        torso_orientation_cost: float = 5.0,
        posture_cost: float = 0.0,
        arm_cost: float = 5.0,  
        lm_damping: float = 0.01,
    ):
        self.model = model
        self.data = data
        self.configuration = mink.Configuration(model)
        
        # ----------------------------------------------------------- #
        # 1. Task 정의 (cost를 파라미터로 받음)
        # ----------------------------------------------------------- #
        
        # (1) CoM Task
        self.com_task = mink.ComTask(cost=com_cost)
        
        # (2) Foot Tasks
        self.left_foot_task = mink.FrameTask(
            frame_name="left_foot",
            frame_type="site",
            position_cost=foot_position_cost,
            orientation_cost=foot_orientation_cost,
            lm_damping=lm_damping
        )
        self.right_foot_task = mink.FrameTask(
            frame_name="right_foot",
            frame_type="site",
            position_cost=foot_position_cost,
            orientation_cost=foot_orientation_cost,
            lm_damping=lm_damping
        )
        
        # (3) Pelvis Orientation Task
        self.pelvis_task = mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=pelvis_orientation_cost,
            lm_damping=lm_damping
        )
        
        # (4) Torso Orientation Task
        self.torso_task = mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=torso_orientation_cost,
            lm_damping=lm_damping
        )
        
        # (5) Posture Task
        self.posture_task = mink.PostureTask(model, cost=posture_cost)
        
        # (6) Arm Posture Task 
        self.arm_task = mink.PostureTask(model, cost=arm_cost)
        
        # 팔 관절 인덱스 저장 
        self.arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
        ]
        
        # 모든 task를 리스트로 관리
        self.tasks = [
            self.com_task,
            self.left_foot_task,
            self.right_foot_task,
            self.pelvis_task,
            self.torso_task,
            self.posture_task,
            self.arm_task,
        ]
        
        self.limits = [mink.ConfigurationLimit(model)]
        self.solver = "daqp"
    
    # ========================================================================= #
    # 초기화: keyframe에서 초기 자세 설정
    # ========================================================================= #
    def initialize_from_keyframe(self, keyframe_name: str = "knees_bent"):
        """keyframe에서 초기 자세를 로드하고 task target 설정"""
        self.configuration.update_from_keyframe(keyframe_name)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.arm_task.set_target_from_configuration(self.configuration)  # arm_task도 초기화
        self.pelvis_task.set_target_from_configuration(self.configuration)
        self.torso_task.set_target_from_configuration(self.configuration)
        
        mujoco.mj_forward(self.model, self.data)

    
    # ========================================================================= #
    # IK Step: Task Space → Joint Space
    # ========================================================================= #
    def solve_ik_step(
        self,
        target_com: np.ndarray,       # (3,) 목표 CoM 위치
        target_left_foot: np.ndarray,  # (3,) 목표 왼발 위치
        target_right_foot: np.ndarray, # (3,) 목표 오른발 위치
        dt: float = 0.002,
        damping: float = 1e-1,
    ):

        # CoM 목표 설정
        self.com_task.set_target(target_com)
        
        # 발 목표 설정 
        self.left_foot_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), target_left_foot
            )
        )
        self.right_foot_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), target_right_foot
            )
        )
        
        # QP 기반 IK 풀기 (모든 task를 soft로)
        vel = mink.solve_ik(
            self.configuration,
            self.tasks, 
            dt,
            self.solver,
            damping=damping,
            limits=self.limits,
        )
        
        # Configuration 업데이트 (적분) - 결과는 self.configuration.q에 저장
        self.configuration.integrate_inplace(vel, dt)
    
    # ========================================================================= #
    # Posture Target 업데이트 (팔 스윙 등)
    # ========================================================================= #
    def update_posture_target(self, target_qpos: np.ndarray):
        """PostureTask의 target qpos를 업데이트 (팔 스윙 등에 사용)"""
        self.posture_task.set_target(target_qpos)
    
    # ========================================================================= #
    # 팔 각도 업데이트 (간편 메서드)
    # ========================================================================= #
    def update_arm_angles(self, arm_angles: dict):

        target_qpos = self.configuration.q.copy()
        for joint_name, angle in arm_angles.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qadr = self.model.jnt_qposadr[jid]
            target_qpos[qadr] = angle

        self.arm_task.set_target(target_qpos)
    
    # ========================================================================= #
    # Collision Avoidance 추가
    # ========================================================================= #
    def add_collision_avoidance(self, geom_pairs: list, 
                                 min_distance: float = 0.01,
                                 detection_distance: float = 0.15):
        self.limits.append(
            mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=geom_pairs,
                minimum_distance_from_collisions=min_distance,
                collision_detection_distance=detection_distance,
            )
        )
