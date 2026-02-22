# 발 궤적 왜곡 문제 해결 (Foot Trajectory Distortion Bug Fix)

## 문제 증상

발 궤적이 기대한 부드러운 **사이클로이드(Cycloid)** 곡선이 아니라, **톱니(Sawtooth)** 또는 **번개 모양**처럼 찌그러지고 뒤로 밀리는 현상 발생.

![문제 이미지 예시]
- 궤적이 위쪽으로 뾰족하게 솟음 (Vertical Spike)
- 뒤쪽 대각선 방향으로 밀림
- 착지 지점이 지그재그로 흔들림

---

## 원인 분석

### 원인 1: Moving Reference Problem (가장 유력)

**문제**: 스윙(Swing) 구간 동안 `init_pos`(출발점)가 고정되지 않고 매 틱마다 업데이트됨.

**정상 로직**:
- 스윙이 **시작되는 순간**에만 `init_pos = current_foot_pos`로 저장
- 스윙이 끝날 때까지 이 변수는 **절대 변하면 안 됨**

**버그 코드**:
```python
# play.py - 매 루프마다 실행
init_pos_L = swing_start_pos if swing_start_pos is not None else left_foot_pos
ref_foot_pos_L = layer1.generate_swing_trajectory(phase, init_pos_L, next_footstep)
```

**문제점**:
- `swing_start_pos`가 None이면 현재 발 위치(`left_foot_pos`)를 사용
- 발이 조금 움직이면 `init_pos`도 따라 움직임
- 보간 수식 `ratio * (target - init)`에서 기준점이 계속 바뀜
- 궤적이 찌그러지고 위로 솟음

---

### 원인 2: 목표 지점의 실시간 변동 (Raibert Heuristic Noise)

**문제**: 매 틱마다 `next_footstep`을 재계산하여 목표 위치가 흔들림.

**버그 코드**:
```python
# play.py - 매 루프마다 실행
next_footstep = layer1.Raibert_Heuristic_foot_step_planner(...)
ref_foot_pos_L = layer1.generate_swing_trajectory(phase, init_pos_L, next_footstep)
```

**문제점**:
- Raibert Heuristic: `x_next = x_hip + T/2 * v + k * (v - v_des)`
- 센서 노이즈나 제어 불안정으로 속도(`v`)가 흔들림
- 목표 지점(`x_next`)도 매 틱마다 변동
- 궤적의 끝부분이 지그재그로 흔들림

---

### 원인 3: 좌표계 혼동 (Global vs Local)

**문제**: 로봇 몸체는 앞으로 가는데, 발 궤적은 그 속도를 보상하지 못함.

**증상**:
- 파란 선들이 뒤쪽 대각선 방향으로 밀림
- 로봇은 앞으로 가는데 발의 목표 지점이 따라가지 못함

**해결 필요**:
- 궤적 생성이 World Frame 기준인지 Body Frame 기준인지 명확히
- IK(역기구학)에 넣을 때 좌표 변환 확인

---

## 해결 방법

### 1. 시작 위치 고정 (원인 1 해결)

**수정 전**:
```python
# Layer1.py
def Raibert_Heuristic_foot_step_planner(self, current_foot_pos, ...):
    # Swing 시작 시점의 발 위치 저장
    if swing_leg_index == 0 and self.swing_start_left_pos is None:
        self.swing_start_left_pos = current_foot_pos.copy()
    
    # 초기 위치 사용 - 문제: None이면 current_foot_pos 사용
    init_pos = self.swing_start_left_pos if self.swing_start_left_pos is not None else current_foot_pos
```

**수정 후**:
```python
# Layer1.py - __init__에 추가
self.swing_start_left_pos = None
self.swing_start_right_pos = None

# state_machine에서 리셋
if prev_p > 0.9 and self.p < 0.1:
    self.swing_start_left_pos = None  # 다음 스윙에서 새로 저장

# play.py - 안전한 fallback
swing_start_pos = layer1.get_swing_start_pos(swing_leg_idx)
if swing_start_pos is None:
    swing_start_pos = left_foot_pos.copy()  # 첫 스윙만 사용
ref_foot_pos_L = layer1.generate_swing_trajectory(phase, swing_start_pos, next_footstep)
```

---

### 2. 목표 위치 고정 (원인 2 해결)

**핵심 아이디어**: 스윙 시작 시 목표 위치를 **한 번만** 계산하고 저장.

**수정 전**:
```python
# play.py - 매 틱마다 재계산
next_footstep = layer1.Raibert_Heuristic_foot_step_planner(...)
```

**수정 후**:
```python
# Layer1.py - __init__에 추가
self.target_footstep_left = None
self.target_footstep_right = None

# state_machine에서 리셋
if prev_p > 0.9 and self.p < 0.1:
    self.swing_start_left_pos = None
    self.target_footstep_left = None  # 목표 위치도 리셋

# Raibert_Heuristic_foot_step_planner 수정
def Raibert_Heuristic_foot_step_planner(self, ...):
    # 목표 발자국 위치 계산 (한 번만)
    if swing_leg_index == 0 and self.target_footstep_left is None:
        self.target_footstep_left = self._calculate_footstep(...)
    elif swing_leg_index == 1 and self.target_footstep_right is None:
        self.target_footstep_right = self._calculate_footstep(...)
    
    # 저장된 목표 위치 반환
    return self.target_footstep_left if swing_leg_index == 0 else self.target_footstep_right

def _calculate_footstep(self, torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data=None):
    """실제 발자국 위치 계산 (내부 함수)"""
    # Raibert Heuristic 계산 로직
    ...
```

---

## 수정 전후 비교

### 수정 전 (버그)
```
Tick 0:  init_pos = [0.0, 0.0], target = [0.3, 0.0]  → 궤적 생성
Tick 1:  init_pos = [0.01, 0.0], target = [0.31, 0.0]  → 기준점 변경! (버그)
Tick 2:  init_pos = [0.02, 0.0], target = [0.29, 0.0]  → 목표도 흔들림! (버그)
...
```

### 수정 후 (정상)
```
Tick 0:  init_pos = [0.0, 0.0], target = [0.3, 0.0]  → 궤적 생성 (고정)
Tick 1:  init_pos = [0.0, 0.0], target = [0.3, 0.0]  → 동일한 기준점 사용 ✓
Tick 2:  init_pos = [0.0, 0.0], target = [0.3, 0.0]  → 동일한 목표 사용 ✓
...
Swing 끝: 리셋
Tick N:  init_pos = [0.3, 0.0], target = [0.6, 0.0]  → 새로운 스윙 시작
```

---

## 핵심 원칙

### 1. 스윙 구간 동안 불변(Immutable)
- **시작 위치** (`swing_start_pos`): 스윙 시작 시 한 번만 저장
- **목표 위치** (`target_footstep`): 스윙 시작 시 한 번만 계산
- 스윙이 끝날 때까지 **절대 변경 금지**

### 2. 명확한 리셋 시점
- Phase가 0.9 → 0.1로 넘어갈 때 (스윙 종료)
- 두 변수 모두 `None`으로 리셋
- 다음 스윙에서 새로운 값으로 초기화

### 3. 안전한 Fallback
- 첫 스윙이거나 초기화 안 된 경우에만 현재 위치 사용
- 이후에는 저장된 값 사용

---

## 추가 개선 사항 (선택)

### Low Pass Filter (목표 위치 평활화)
```python
# 목표 위치에 LPF 적용 (노이즈 제거)
alpha = 0.8  # 필터 계수
target_filtered = alpha * target_prev + (1 - alpha) * target_new
```

### Deadband (착지 직전 목표 고정)
```python
# 스윙의 80% 이후에는 목표 위치 수정 금지
if phase > 0.8:
    next_footstep = self.target_footstep_locked
```

---

## 결론

발 궤적 왜곡 문제는 **Moving Reference**와 **목표 위치 변동**이 주요 원인이었습니다.

**해결 핵심**:
1. 스윙 시작 시 시작 위치와 목표 위치를 **한 번만** 저장
2. 스윙 구간 동안 **절대 변경하지 않음**
3. 스윙 종료 시 명확하게 리셋

이제 발 궤적이 부드러운 사이클로이드 곡선을 그릴 것입니다! 🚀
