"""ZMP Preview + 실시간 IK + mj_step 물리 시뮬레이션

오프라인 IK 없이, 매 시뮬레이션 스텝마다:
  1. Cartesian 궤적에서 현재 시점의 목표 읽기
  2. KinematicWBC.solve() → 목표 관절각
  3. data.ctrl = q_target[actuator] → mj_step

play.py: 오프라인 IK → mj_fwdPosition (재생만)
play_sim.py: 오프라인 IK → CubicSpline → mj_step
play_realtime.py: 실시간 IK → mj_step (이 파일)
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

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
    model.opt.timestep = DT

    # 초기 상태
    mujoco.mj_resetData(model, data)
    key_id = model.key(name="knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # ── actuator 매핑 ──
    actuator_dof_ids = []
    actuator_qpos_ids = []
    for name in ACTUATOR_NAMES:
        jnt_id = model.actuator(name).trnid[0]
        actuator_dof_ids.append(model.jnt_dofadr[jnt_id])
        actuator_qpos_ids.append(model.jnt_qposadr[jnt_id])
    actuator_dof_ids = np.array(actuator_dof_ids)
    actuator_qpos_ids = np.array(actuator_qpos_ids)

    upper_body_dof_ids = []
    for name in UPPER_BODY_ACTUATOR_NAMES:
        jnt_id = model.actuator(name).trnid[0]
        upper_body_dof_ids.append(model.jnt_dofadr[jnt_id])
    upper_body_dof_ids = np.array(upper_body_dof_ids)

    # ── 초기 위치 ──
    init_com = data.subtree_com[0].copy()
    init_lf = data.site(LEFT_FOOT_SITE).xpos.copy()
    init_rf = data.site(RIGHT_FOOT_SITE).xpos.copy()

    print(f"[초기 상태] CoM={init_com}, LF={init_lf}, RF={init_rf}")

    # ── Cartesian 궤적 생성 (오프라인) ──
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
    sim_end = L * DT
    print(f"[궤적 생성 완료] {L} samples, {sim_end:.2f}s")

    # ── IK 엔진 (실시간용 — 별도 model/data 사용) ──
    # mj_step이 data를 변경하므로, IK는 별도 data 복사본에서 수행
    ik_data = mujoco.MjData(model)

    wbc = KinematicWBC(model, ik_data, actuator_dof_ids)

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

    # ── 시뮬레이션 ──
    # 초기 ctrl
    data.ctrl[:] = data.qpos[actuator_qpos_ids]
    simfreq = 60

    print(f"\n[실시간 시뮬레이션 시작] {sim_end:.2f}s")

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data,
            show_left_ui=True, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.flags[16] = 1
            viewer.opt.flags[18] = 1

            time.sleep(1)
            print("  시뮬레이션 중...")

            while viewer.is_running() and data.time < sim_end:
                clock_start = time.time()
                time_prev = data.time

                while (data.time - time_prev < 1.0 / simfreq) and data.time < sim_end:
                    k = min(int(data.time / DT), L - 1)

                    # 현재 시점의 Cartesian 목표
                    ref_com = np.array([
                        traj['ref_com_pos_x'][k],
                        traj['ref_com_pos_y'][k],
                        init_com[2],
                    ])
                    ref_lf = traj['left_foot_traj'][k]
                    ref_rf = traj['right_foot_traj'][k]

                    # IK: sim data → ik_data 복사 후 풀기
                    ik_data.qpos[:] = data.qpos.copy()
                    ik_data.qvel[:] = data.qvel.copy()

                    q_target, _ = wbc.solve(
                        ref_com, ref_lf, ref_rf,
                        upper_body_dof_ids=upper_body_dof_ids,
                    )

                    # position actuator: ctrl = 목표 관절각
                    data.ctrl[:] = q_target[actuator_qpos_ids]

                    mujoco.mj_step(model, data)

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
