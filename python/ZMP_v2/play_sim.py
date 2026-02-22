"""ZMP Preview + Kinematic WBC + mj_step 물리 시뮬레이션

ctrl(lib_ZMPctrl.py)의 전체 파이프라인 재현:
  1. 오프라인 Cartesian 궤적 (TrajectoryPlanner)
  2. 오프라인 IK → qtraj (KinematicWBC, cart2joint)
  3. CubicSpline 보간 (ctrl의 qspl)
  4. 실시간: data.ctrl = qspl(t)[6:] → mj_step (position actuator)

position actuator (kp=10000, kv=100) → data.ctrl에 목표 관절각만 넣으면
MuJoCo 내부 PD가 추종. ctrl의 posCTRL=True 모드와 동일.
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DT, COM_HEIGHT, N_STEPS, STEP_LENGTH, STEP_WIDTH, STEP_HEIGHT,
    STEP_TIME, DSP_RATIO, PREVIEW_HORIZON,
    ACTUATOR_NAMES, UPPER_BODY_ACTUATOR_NAMES,
    LEFT_FOOT_SITE, RIGHT_FOOT_SITE,
)
from trajectory_planner import TrajectoryPlanner
from kinematic_wbc import KinematicWBC


def main():
    # ── 모델 로드 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(script_dir, "../g1/scene_29dof_pos.xml"))

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep  # XML에서 설정된 시뮬레이션 timestep

    # 초기 상태 — knees_bent keyframe
    mujoco.mj_resetData(model, data)
    key_id = model.key(name="knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # ── actuator 매핑 ──
    actuator_qpos_ids = []
    for name in ACTUATOR_NAMES:
        jnt_id = model.actuator(name).trnid[0]
        qpos_id = model.jnt_qposadr[jnt_id]
        actuator_qpos_ids.append(qpos_id)
    actuator_qpos_ids = np.array(actuator_qpos_ids)

    actuator_dof_ids = []
    for name in ACTUATOR_NAMES:
        jnt_id = model.actuator(name).trnid[0]
        dof_id = model.jnt_dofadr[jnt_id]
        actuator_dof_ids.append(dof_id)
    actuator_dof_ids = np.array(actuator_dof_ids)

    upper_body_dof_ids = []
    for name in UPPER_BODY_ACTUATOR_NAMES:
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

    # ── 2단계: 오프라인 IK (cart2joint) ──
    wbc = KinematicWBC(model, data, actuator_dof_ids)

    qtraj = np.zeros((L, model.nq))
    ttraj = np.arange(L) * DT
    print(f"\n[IK 시작] {L} steps...")

    for k in range(L):
        ref_com = np.array([
            traj['ref_com_pos_x'][k],
            traj['ref_com_pos_y'][k],
            init_com[2],
        ])
        ref_lf = traj['left_foot_traj'][k]
        ref_rf = traj['right_foot_traj'][k]

        q_cmd, _ = wbc.solve(
            ref_com, ref_lf, ref_rf,
            upper_body_dof_ids=upper_body_dof_ids,
        )
        qtraj[k] = q_cmd

        if k % 500 == 0:
            err = wbc.compute_total_error(ref_com, ref_lf, ref_rf)
            print(f"  step {k}/{L}, err={err:.6e}")

    print(f"[IK 완료]")

    # ── 3단계: CubicSpline 보간 (ctrl의 qspl) ──
    # ctrl: CubicSpline(ttraj, qtraj[:, i]) — 각 DoF별
    print(f"[CubicSpline 보간 생성]")
    qspl = {}
    for i, qpos_id in enumerate(actuator_qpos_ids):
        qspl[i] = CubicSpline(ttraj, qtraj[:, qpos_id])

    sim_end = ttraj[-1]

    # ── 궤적 시각화 데이터 ──
    vis_step = max(1, int(0.02 / DT))
    com_traj_3d = np.column_stack([
        traj['ref_com_pos_x'], traj['ref_com_pos_y'],
        np.full(L, init_com[2])
    ])
    zmp_traj_3d = np.column_stack([
        traj['zmp_actual_x'][:L], traj['zmp_actual_y'][:L],
        np.full(L, init_lf[2])
    ])
    lf_traj_3d = traj['left_foot_traj']
    rf_traj_3d = traj['right_foot_traj']

    footstep_positions = []
    for i, (fx, fy) in enumerate(traj['footsteps']):
        gz = init_lf[2] if i % 2 == 0 else init_rf[2]
        footstep_positions.append((fx, fy, gz, i % 2 == 0))

    # ── 4단계: mj_step 시뮬레이션 ──
    # 초기 상태: IK 결과의 첫 프레임 (ctrl: data=q2data(data, humn.q0))
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qtraj[0]
    data.ctrl[:] = qtraj[0][actuator_qpos_ids]
    mujoco.mj_forward(model, data)

    print(f"\n[초기 상태 확인]")
    print(f"  sim CoM:  {data.subtree_com[0]}")
    print(f"  ref CoM:  [{traj['ref_com_pos_x'][0]:.6f}, {traj['ref_com_pos_y'][0]:.6f}, {init_com[2]:.6f}]")
    print(f"  sim LF:   {data.site(LEFT_FOOT_SITE).xpos}")
    print(f"  sim RF:   {data.site(RIGHT_FOOT_SITE).xpos}")

    print(f"\n[시뮬레이션 시작] {sim_end:.2f}s, dt={sim_dt}")
    print("  viewer 창을 닫으면 종료됩니다.")

    simfreq = 60

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=True, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.flags[16] = 1  # Contact Forces
            viewer.opt.flags[18] = 1  # Transparent

            time.sleep(1)
            print("  시뮬레이션 중...")

            log_interval = 0.5
            next_log_time = 0.0

            while viewer.is_running() and data.time < sim_end:
                clock_start = time.time()
                time_prev = data.time

                while (data.time - time_prev < 1.0 / simfreq) and data.time < sim_end:
                    t_now = data.time
                    for i in range(model.nu):
                        if t_now <= sim_end:
                            data.ctrl[i] = qspl[i](t_now)

                    mujoco.mj_step(model, data)

                    # 로그
                    if data.time >= next_log_time:
                        k_now = min(int(data.time / DT), L - 1)
                        sim_com = data.subtree_com[0].copy()
                        ref_com_now = np.array([traj['ref_com_pos_x'][k_now], traj['ref_com_pos_y'][k_now], init_com[2]])
                        com_err = np.linalg.norm(sim_com - ref_com_now)
                        print(f"  t={data.time:.3f} | CoM_err={com_err:.4f} | sim_com={sim_com}")
                        if com_err > 0.5:
                            print(f"    *** CoM 발산!")
                        next_log_time = data.time + log_interval

                if data.time >= sim_end:
                    break

                # ── 궤적 시각화 ──
                k_vis = min(int(data.time / DT), L - 1)
                idx_geom = 0
                max_geom = viewer.user_scn.maxgeom - 10

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

                for j in range(k_vis, L, vis_step):
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

                for j in range(k_vis, L, vis_step):
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

                for j in range(k_vis, L, vis_step):
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

                for j in range(k_vis, L, vis_step):
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

                if idx_geom < max_geom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=com_traj_3d[k_vis],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 1, 0, 0.8], dtype=np.float32),
                    )
                    idx_geom += 1

                viewer.user_scn.ngeom = idx_geom
                viewer.sync()

                elapsed = time.time() - clock_start
                sleep_time = 1.0 / simfreq - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"[시뮬레이션 완료] t={data.time:.3f}s")
    except Exception as e:
        print(f"[viewer 종료] {e}")


if __name__ == "__main__":
    main()
