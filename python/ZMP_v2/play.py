"""ZMP Preview + Kinematic WBC 시각화 (mj_fwdPosition only, no mj_step)

ctrl(lib_ZMPctrl.py)의 cart2joint 방식:
  1. 오프라인 Cartesian 궤적 생성 (TrajectoryPlanner)
  2. 매 dt마다 KinematicWBC.solve → qtraj 배열
  3. viewer에서 qpos 직접 설정 + mj_fwdPosition → 시각화

mj_step 없음 → 물리 시뮬레이션 없이 IK 결과만 확인.
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DT, COM_HEIGHT, N_STEPS, STEP_LENGTH, STEP_WIDTH, STEP_HEIGHT,
    STEP_TIME, DSP_RATIO, PREVIEW_HORIZON,
    ACTUATOR_NAMES, UPPER_BODY_ACTUATOR_NAMES,
    LEFT_FOOT_SITE, RIGHT_FOOT_SITE,
)
from trajectory_planner import TrajectoryPlanner
from kinematic_wbc import KinematicWBC


def plot_trajectories(traj, L, dt, init_com):
    """IK 전에 오프라인 Cartesian 궤적을 시각화."""
    t = np.arange(L) * dt

    com_x = traj['ref_com_pos_x']
    com_y = traj['ref_com_pos_y']
    zmp_x = traj['zmp_actual_x']
    zmp_y = traj['zmp_actual_y']
    lf = traj['left_foot_traj']
    rf = traj['right_foot_traj']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Offline Cartesian Trajectories (before IK)')

    # ── X 궤적 (시간) ──
    ax = axes[0, 0]
    ax.plot(t, com_x, 'b', label='CoM x')
    ax.plot(t, zmp_x, 'r--', label='ZMP x')
    ax.plot(t, lf[:, 0], 'g', alpha=0.7, label='LF x')
    ax.plot(t, rf[:, 0], 'm', alpha=0.7, label='RF x')
    ax.set_ylabel('x [m]')
    ax.set_title('X (forward)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Y 궤적 (시간) ──
    ax = axes[0, 1]
    ax.plot(t, com_y, 'b', label='CoM y')
    ax.plot(t, zmp_y, 'r--', label='ZMP y')
    ax.plot(t, lf[:, 1], 'g', alpha=0.7, label='LF y')
    ax.plot(t, rf[:, 1], 'm', alpha=0.7, label='RF y')
    ax.set_ylabel('y [m]')
    ax.set_title('Y (lateral)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Z 궤적 (발 높이) ──
    ax = axes[1, 0]
    ax.plot(t, lf[:, 2], 'g', label='LF z')
    ax.plot(t, rf[:, 2], 'm', label='RF z')
    ax.set_ylabel('z [m]')
    ax.set_title('Foot Z (swing height)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── XY 평면 (top view) ──
    ax = axes[1, 1]
    ax.plot(com_x, com_y, 'b', label='CoM')
    ax.plot(zmp_x[:L], zmp_y[:L], 'r--', alpha=0.5, label='ZMP')
    ax.plot(lf[:, 0], lf[:, 1], 'g', alpha=0.7, label='LF')
    ax.plot(rf[:, 0], rf[:, 1], 'm', alpha=0.7, label='RF')
    # footstep 위치 마커
    for i, (fx, fy) in enumerate(traj['footsteps']):
        color = 'g' if i % 2 == 0 else 'm'
        ax.plot(fx, fy, 's', color=color, markersize=8, alpha=0.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('XY plane (top view)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── CoM 속도 ──
    ax = axes[2, 0]
    ax.plot(t, traj['ref_com_vel_x'], 'b', label='CoM vx')
    ax.plot(t, traj['ref_com_vel_y'], 'r', label='CoM vy')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('vel [m/s]')
    ax.set_title('CoM velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Phase info ──
    ax = axes[2, 1]
    phase_names = [p[0] for p in traj['phase_info']]
    swing_ids = [p[1] for p in traj['phase_info']]
    # DSP=0, SSP_left_swing=1, SSP_right_swing=2
    phase_val = np.array([0 if s == -1 else (1 if s == 0 else 2) for s in swing_ids])
    ax.plot(t, phase_val, 'k', linewidth=0.5)
    ax.fill_between(t, phase_val, alpha=0.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['DSP', 'SSP (LF swing)', 'SSP (RF swing)'])
    ax.set_xlabel('time [s]')
    ax.set_title('Gait phase')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectories.png', dpi=150)
    print("[플롯 저장] trajectories.png")
    plt.show(block=False)
    plt.pause(0.5)


def main():
    # ── 모델 로드 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(script_dir, "../g1/scene_29dof_pos.xml"))

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    # 초기 상태 — knees_bent keyframe 사용
    mujoco.mj_resetData(model, data)
    key_id = model.key(name="knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # ── actuator DoF ID 매핑 ──
    actuator_dof_ids = []
    for name in ACTUATOR_NAMES:
        act_id = model.actuator(name).id
        jnt_id = model.actuator(name).trnid[0]
        dof_id = model.jnt_dofadr[jnt_id]
        actuator_dof_ids.append(dof_id)
    actuator_dof_ids = np.array(actuator_dof_ids)

    # 상체 DoF IDs (WBC 2순위 잠금)
    upper_body_dof_ids = []
    for name in UPPER_BODY_ACTUATOR_NAMES:
        act_id = model.actuator(name).id
        jnt_id = model.actuator(name).trnid[0]
        dof_id = model.jnt_dofadr[jnt_id]
        upper_body_dof_ids.append(dof_id)
    upper_body_dof_ids = np.array(upper_body_dof_ids)

    # ── 초기 위치 읽기 ──
    init_com = data.subtree_com[0].copy()
    init_lf = data.site(LEFT_FOOT_SITE).xpos.copy()
    init_rf = data.site(RIGHT_FOOT_SITE).xpos.copy()

    print(f"[초기 상태]")
    print(f"  CoM:  {init_com}")
    print(f"  LF:   {init_lf}")
    print(f"  RF:   {init_rf}")

    # ── 1단계: 오프라인 Cartesian 궤적 생성 ──
    planner = TrajectoryPlanner(
        z_c=COM_HEIGHT, dt=DT, step_time=STEP_TIME,
        dsp_ratio=DSP_RATIO, step_height=STEP_HEIGHT,
        preview_horizon=PREVIEW_HORIZON,
    )
    traj = planner.compute_all_trajectories(
        n_steps=N_STEPS, step_length=STEP_LENGTH, step_width=STEP_WIDTH,
        init_com=init_com, init_lf=init_lf, init_rf=init_rf,
    )
    L = traj['length']
    print(f"\n[궤적 생성 완료] {L} samples, {L * DT:.2f}s")

    # ── 궤적 플롯 ──
    plot_trajectories(traj, L, DT, init_com)

    # ── 2단계: 오프라인 IK (cart2joint) ──
    wbc = KinematicWBC(model, data, actuator_dof_ids)

    qtraj = np.zeros((L, model.nq))
    print(f"\n[IK 시작] {L} steps...")

    for k in range(L):
        # 목표 Cartesian 위치
        ref_com = np.array([
            traj['ref_com_pos_x'][k],
            traj['ref_com_pos_y'][k],
            init_com[2],  # CoM z 고정 (LIPM 가정)
        ])
        ref_lf = traj['left_foot_traj'][k]
        ref_rf = traj['right_foot_traj'][k]

        # IK 풀기
        q_cmd, _ = wbc.solve(
            ref_com, ref_lf, ref_rf,
            upper_body_dof_ids=upper_body_dof_ids,
        )
        qtraj[k] = q_cmd

        if k % 500 == 0:
            cur_com = data.subtree_com[0].copy()
            cur_lf = data.site(LEFT_FOOT_SITE).xpos.copy()
            cur_rf = data.site(RIGHT_FOOT_SITE).xpos.copy()
            err = wbc.compute_total_error(ref_com, ref_lf, ref_rf)
            print(f"  step {k}/{L}, err={err:.6e}")
            print(f"    ref_com={ref_com}, cur_com={cur_com}, d={np.linalg.norm(ref_com-cur_com):.4e}")
            print(f"    ref_lf={ref_lf}, cur_lf={cur_lf}, d={np.linalg.norm(ref_lf-cur_lf):.4e}")
            print(f"    ref_rf={ref_rf}, cur_rf={cur_rf}, d={np.linalg.norm(ref_rf-cur_rf):.4e}")

    print(f"[IK 완료]")

    # ── 3단계: viewer에서 재생 (mj_fwdPosition only) ──
    print(f"\n[재생 시작] {L * DT:.2f}s at {1/DT:.0f}Hz")
    print("  viewer 창을 닫으면 종료됩니다.")

    playback_speed = 1.0  # 1.0 = 실시간

    # 궤적 시각화용 데이터 준비 (매 프레임 그리면 느리니까 서브샘플)
    vis_step = max(1, int(0.02 / DT))  # 20ms 간격으로 궤적 점 표시
    com_traj_3d = np.column_stack([
        traj['ref_com_pos_x'], traj['ref_com_pos_y'],
        np.full(L, init_com[2])
    ])
    zmp_traj_3d = np.column_stack([
        traj['zmp_actual_x'][:L], traj['zmp_actual_y'][:L],
        np.full(L, init_lf[2])  # 지면 높이
    ])
    lf_traj_3d = traj['left_foot_traj']
    rf_traj_3d = traj['right_foot_traj']

    # footstep 위치 (지면 높이)
    footstep_positions = []
    for i, (fx, fy) in enumerate(traj['footsteps']):
        gz = init_lf[2] if i % 2 == 0 else init_rf[2]
        footstep_positions.append((fx, fy, gz, i % 2 == 0))  # (x,y,z, is_left)

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=True, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.flags[16] = 1  # Contact Forces
            viewer.opt.flags[18] = 1  # Transparent

            time.sleep(1)
            print("  재생 중...")

            k = 0
            t_start = time.time()

            while viewer.is_running() and k < L:
                # qpos 직접 설정 (mj_step 없음)
                data.qpos[:] = qtraj[k]
                mujoco.mj_fwdPosition(model, data)

                # ── user_scn에 궤적 그리기 ──
                idx_geom = 0
                max_geom = viewer.user_scn.maxgeom - 10

                # (1) footstep 목표 위치 — 사각형 마커
                for fx, fy, fz, is_left in footstep_positions:
                    if idx_geom >= max_geom:
                        break
                    color = [0, 1, 0, 0.4] if is_left else [1, 0, 1, 0.4]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.04, 0.025, 0.001],
                        pos=np.array([fx, fy, fz]),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(color, dtype=np.float32),
                    )
                    idx_geom += 1

                # (2) CoM 궤적 — 파란 점 (미래 궤적)
                for j in range(k, L, vis_step):
                    if idx_geom >= max_geom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.005, 0, 0],
                        pos=com_traj_3d[j],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 1, 0.3], dtype=np.float32),
                    )
                    idx_geom += 1

                # (3) 왼발 궤적 — 초록 점
                for j in range(k, L, vis_step):
                    if idx_geom >= max_geom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.003, 0, 0],
                        pos=lf_traj_3d[j],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 1, 0, 0.3], dtype=np.float32),
                    )
                    idx_geom += 1

                # (4) 오른발 궤적 — 마젠타 점
                for j in range(k, L, vis_step):
                    if idx_geom >= max_geom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.003, 0, 0],
                        pos=rf_traj_3d[j],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 1, 0.3], dtype=np.float32),
                    )
                    idx_geom += 1

                # (5) ZMP 궤적 — 빨간 점
                for j in range(k, L, vis_step):
                    if idx_geom >= max_geom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.004, 0, 0],
                        pos=zmp_traj_3d[j],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 0.4], dtype=np.float32),
                    )
                    idx_geom += 1

                # (6) 현재 목표 CoM — 큰 노란 구
                if idx_geom < max_geom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=com_traj_3d[k],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 1, 0, 0.8], dtype=np.float32),
                    )
                    idx_geom += 1

                viewer.user_scn.ngeom = idx_geom
                viewer.sync()

                k += 1

                # 실시간 동기화
                t_elapsed = time.time() - t_start
                t_sim = k * DT / playback_speed
                if t_sim > t_elapsed:
                    time.sleep(t_sim - t_elapsed)

            print(f"[재생 완료] {k} frames")
    except Exception as e:
        print(f"[viewer 종료] {e}")


if __name__ == "__main__":
    main()
