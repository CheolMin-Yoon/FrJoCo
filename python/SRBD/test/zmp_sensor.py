import numpy as np
import mujoco

FZ_THRESHOLD = 1.0  # (N)

def compute_zmp_from_ft_sensors(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:

    # 1. 발 Body ID 조회
    lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    
    # site가 부착된 body를 찾습니다.
    lf_body_id = model.site_bodyid[lf_site_id]
    rf_body_id = model.site_bodyid[rf_site_id]

    # 발에 해당하는 Body ID 집합 (발이 여러 개의 Geom으로 쪼개져 있을 경우를 대비해 확장 가능)
    foot_body_ids = {lf_body_id, rf_body_id}

    total_fz = 0.0
    weighted_pos = np.zeros(2)
    
    # MuJoCo의 contact 배열은 미리 할당된 버퍼이므로 ncon까지만 순회해야 함
    for i in range(data.ncon):
        # 2. 접촉 관련 Body 확인 (최적화를 위해 힘 계산 전에 먼저 체크)
        # contact geom이 속한 body ID를 가져옵니다.
        g1_body = model.geom_bodyid[data.contact[i].geom1]
        g2_body = model.geom_bodyid[data.contact[i].geom2]

        # 둘 중 하나라도 발이면 계산 포함
        if g1_body not in foot_body_ids and g2_body not in foot_body_ids:
            continue

        # 3. 접촉력 계산
        fci = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, fci)
        
        # fci[0]는 Contact Frame의 Normal Force (항상 양수여야 함)
        # 하지만 미끄러짐 등 불안정한 상황 대비 abs 사용은 유지
        fn = abs(fci[0])
        
        if fn < FZ_THRESHOLD:
            continue

        # 4. 접촉 위치 (World Frame)
        contact_pos = data.contact[i].pos

        # 가중 합산
        total_fz += fn
        weighted_pos += contact_pos[:2] * fn

    # 5. 결과 반환
    if total_fz > FZ_THRESHOLD:
        zmp = weighted_pos / total_fz
    else:
        # 공중에 있을 때: 양발 Site의 중점
        lf_pos = data.site_xpos[lf_site_id]
        rf_pos = data.site_xpos[rf_site_id]
        zmp = (lf_pos[:2] + rf_pos[:2]) / 2.0

    return zmp