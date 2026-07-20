# RGBD 스터드 타이어 홀 검사 (RealSense D435)

3D 프린트한 스터드/스파이크 타이어 표면 모형(검은색 PLA)에서, 실제로 스터드 핀(압정)을
박기 전에 설계된 홀이 제대로 뚫려 있는지 RealSense D435로 검사하는 프로젝트.

## 왜 Depth 기반인가

출력물이 무광 검은색 플라스틱이라 RGB 카메라만으로는 홀과 주변 표면의 대비가 거의 없다.
대신 D435의 depth 채널로 "주변 표면보다 국소적으로 더 멀리(더 깊이) 측정되는 영역"을 찾는
방식을 쓴다. RGB는 시각화용으로만 사용한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `model.py` | 메인 실행 스크립트. D435 캡처 -> 정렬(align) -> 홀 탐지 -> 화면 표시 -> `s`로 결과 저장 |
| `hole_detector.py` | 핵심 탐지 알고리즘 (`HoleDetector`, `HoleDetectorConfig`) |
| `calibration.py` | 카메라 좌표 -> 로봇 베이스 좌표 변환 (`hand_eye_calibration.json` 로드) |
| `hand_eye_calibration.example.json` | 핸드아이 캘리브레이션 결과 파일 형식 예시 + 동차좌표/캘리브레이션 절차 설명 |
| `requirements.txt` | 의존 패키지 목록 |

## 설치 및 실행

```bash
pip install -r requirements.txt
python model.py
```

조작: `q` 종료, `s` 현재 프레임 홀 좌표를 `holes_<timestamp>.json`으로 저장, 미리보기 창
클릭 시 그 지점의 실제 depth(m/mm)를 콘솔에 출력 (거리 튜닝용 디버그 도구).

## 탐지 알고리즘 개요

1. **물체 분리** (`segment_object`): 설정된 거리 범위(`min/max_object_distance_m`) 안에서
   가장 큰 연결 성분을 타이어 출력물로 간주하고 배경(책상, 벽, 손 등)을 제외한다.
2. **baseline 표면 추정**: 예상 홀 지름보다 큰 커널로 depth map에 grayscale morphological
   opening을 적용해 "홀이 없었다면 이랬을 것"이라는 매끈한 기준 표면을 복원한다. 절대
   높이가 아니라 국소(local) 기준이라 타이어 표면이 곡면이거나 기울어져 있어도 동작한다.
3. **잔차(residual)로 홀 마스크 생성**: 실측 depth - baseline이 임계값(`min_hole_depth_mm`)
   이상인 픽셀만 홀 후보로 남긴다. 깊은 홀에서 흔한 depth 측정 실패(dropout, 값=0)도 강한
   홀 신호로 취급한다.
4. **개별 홀 추출** (`_extract_holes`): 후보 블롭을 면적(설계된 홀 지름 범위)과 원형도
   (`4·π·area/perimeter²`)로 필터링해 노이즈/표면 결함/트레드 문양을 제외하고, 살아남은
   블롭마다 중심 픽셀 -> 카메라 3D 좌표(`rs2_deproject_pixel_to_point`)까지 계산한다.
5. **로봇 좌표 변환** (`calibration.py`): 4x4 동차 변환 행렬로 카메라 좌표를 로봇 베이스
   좌표로 옮긴다. 캘리브레이션 파일이 없으면 identity로 대체하고 경고를 띄운다.

## 실제 부품 스펙

- 홀(스터드 핀 머리 입구): 지름 11mm, 깊이 3mm — **카메라가 찾아야 하는 대상**
- 핀 끝부분: 지름 2mm, 깊이 7mm — 입구 안쪽 깊은 곳이라 D435 해상도 밖, 비전 탐지 대상 아님
  (로봇이 11mm 입구 중심만 정확히 맞추면 됨)
- 타이어 표면 문양이 V자 형태라, 홀뿐 아니라 트레드 문양 자체에도 depth 굴곡이 있다 ->
  원형도 필터가 "진짜 홀 vs 트레드 문양"을 가르는 핵심 파라미터

## 알려진 이슈 / 다음 단계

- **작업 거리 튜닝 중**: 첫 테스트에서 타이어 모형이 벽에 기대어 카메라로부터 약 1m 거리에
  있었는데, D435는 거리가 멀수록 depth 노이즈가 커져서(1m에서 수 mm 수준) 3mm짜리 홀
  깊이와 노이즈가 구분이 잘 안 되는 문제가 있었다. 실제 검사 때는 카메라를 0.3~0.5m로 더
  가깝게 두는 걸 권장. 클릭 depth 디버그 도구로 정확한 거리를 재서
  `min/max_object_distance_m`을 좁게 맞추는 작업이 진행 중.
- **`min_circularity`, `baseline_kernel_px` 실측 튜닝 필요**: V자 트레드 문양이 홀로
  오탐지되지 않는지 실제 캡처 데이터로 확인 후 조정해야 한다.
- **핸드아이 캘리브레이션 미완료**: `hand_eye_calibration.json`이 아직 없어 로봇 좌표
  변환이 identity(카메라 좌표 그대로)로 동작 중. `hand_eye_calibration.example.json`에
  절차 설명 있음.
