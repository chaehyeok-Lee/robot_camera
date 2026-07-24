"""[역할] RGB Hough Circle + Depth 국소 링(ring) 검증 기반 홀 탐지.

기존(v1) 방식은 이미지 전체에 morphological opening으로 baseline 표면을 추정한
뒤 잔차(residual)로 홀을 찾았는데, 이 물체(V자 트레드 문양)에서는 baseline
추정 커널이 홀 옆의 V자 띠까지 걸쳐서 baseline 자체가 오염되는 근본 문제가
있었다 (README "알려진 근본 문제" 참고).

이 버전은 순서를 바꾼다:
1. RGB(CLAHE 보정) + depth 국소 굴곡(depression) 이미지 양쪽에서 Hough Circle로
   "원형 후보"를 먼저 찾는다 (baseline 오염과 무관하게 동작).
2. 각 후보마다 이미지 전체가 아니라 그 후보를 감싸는 **고리(ring) 영역**의 depth만
   참조값으로 써서 실제로 파여있는지 검증한다 (커널이 옆으로 새는 문제가 없음).
3. 후보 주변을 8방향으로 나눠 "모양이 둥근가"(=8방향이 서로 고르게 깊은가)를
   확인한다. 절대 깊이 크기가 아니라 방향별 균일성(변동계수)을 본다 - 실측 결과
   진짜 홀도 residual이 0~4.5mm로 크게 흔들려서 "몇 mm 이상"이라는 절대 크기
   기준은 근본적으로 불안정했다. 반면 진짜 둥근 홀은 신호 세기와 무관하게 사방에서
   고르게 깊고, 트레드 홈(V자 띠)은 홈이 지나가는 한두 방향만 깊어 편차가 커서
   이 차이로 구분한다.
4. 후보의 깊은 영역이 반경 밖 그루브까지 이어지면(=홈의 일부) 제외한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HoleDetectorConfig:
    """[역할] 탐지 파라미터 모음.

    스터드 핀(압정) 스펙: 머리 지름 11mm / 깊이 3mm (= 카메라가 찾아야 하는
    입구 원). 실측(다중 프레임 평균)으로는 depth 양자화 + 부분 부피 효과 때문에
    실제 residual이 1~2mm 정도로 관측된다.
    """

    # 물체(타이어 출력물) 분리용 거리 범위. distance_calibration.py 실측 결과
    # 294mm보다 가까우면 D435 스테레오 시야 오버랩이 부족해져 유효 depth 자체가
    # 급감했다(물리적 한계, 튜닝으로 해결 불가) - 하한을 그 실측값 기준으로 올림.
    # 상한은 패널 좌우 끝이 중앙보다 살짝 멀 수 있어 여유를 뒀다.
    min_object_distance_m: float = 0.28
    max_object_distance_m: float = 0.55

    # 트레드에서 가장 넓은 채널의 폭(mm) - tire_chevron_v4.stl 기준 V자/중앙
    # 채널이 5mm로 가장 넓음. segment_object의 Closing 커널이 이보다 확실히
    # 커야 채널 때문에 블록들이 서로 다른 섬으로 갈라지지 않는다.
    max_channel_width_mm: float = 6.0

    # 핀 머리가 들어가는 입구 지름 11mm 기준, 프린팅 오차 대비 여유 (탐지 범위를
    # 넓히기 위해 살짝 더 여유를 둠 - 원 판정 자체는 hough_param2로 더 엄격해짐)
    min_hole_diameter_mm: float = 8.0
    max_hole_diameter_mm: float = 14.0

    # 후보 중심이 주변 링보다 몇 mm 더 깊어야(=멀어야) "진짜 홀"로 인정할지.
    # 실측 결과 진짜 홀인데도 최대 4.5mm까지 나오는 경우가 있어서 상한을 올림 -
    # 채널(V자 홈)은 10mm급이라 6mm 이하로 잡아도 여전히 확실히 구분된다.
    min_hole_depth_mm: float = 0.8
    max_hole_depth_mm: float = 6.0

    # Hough Circle 파라미터 (Canny 상위 임계값 / 누산기 임계값 - 낮을수록 후보가 늘어남).
    # param2를 올릴수록 "더 원에 가까운 모양"만 후보로 인정한다(더 엄격). 근본
    # 버그(물체 분리/블러 sigma/필터)를 다 고친 뒤로는 대부분의 진짜 홀이 안정적인
    # 원으로 검출돼서, 정확도를 위해 더 엄격하게 올렸다 - 대신 크기/거리 범위를
    # 넓혀서(위 min/max_hole_diameter_mm, min/max_object_distance_m) 놓치는
    # 홀이 없도록 보완했다.
    hough_param1: float = 10.0
    hough_param2: float = 11.0

    # 후보 원 중심 사이 최소 간격(mm) - px는 매 프레임 거리로 자동 환산.
    # 홀 하나 주변에 잡음으로 여러 후보가 겹쳐서 중복/오탐지로 이어지는 걸 막기
    # 위해 여유를 좀 더 뒀다 (실제 홀 간격은 리브 하나 폭 수준이라 25mm+로 안전).
    min_candidate_distance_mm: float = 20.0

    # 국소 표면(baseline) 추정용 가우시안 블러의 sigma(px). 0이면 최대 홀 반지름의
    # 약 2.5배로 매 프레임 자동 계산 - 실측 검증 결과 sigma가 홀 반지름의 최소
    # 2~2.5배는 돼야 블러가 홀을 확실히 "지우고" 진짜 배경 표면을 복원한다
    # (sigma가 반지름과 비슷한 정도면 홀 자신의 값이 local_surface에 새어 들어가
    # residual이 실제보다 훨씬 작게 나옴).
    surface_blur_sigma_px: float = 0.0

    # 양방향 필터(bilateral)의 sigma_color(mm) - 이보다 값 차이가 작은 경계는
    # 지우고(=홀), 크면 보존한다(=채널). 실측+합성 테스트로 3.0mm 정도가 "홀
    # 신호는 잘 지우면서 7mm 클리어런스 밖 채널 신호가 안 새어 들어오는" 균형점.
    depression_sigma_color_mm: float = 3.0

    # 후보의 깊은 영역이 이 mm 임계값 기준으로 반경 밖까지 이어지면 그루브(홈)의
    # 일부로 보고 제외한다. 검사 반경(반지름의 1.75배)이 설계상 홀-채널 최소
    # 클리어런스(7mm)보다 넓어서, 모든 진짜 홀이 이 반경 안에 근처 채널을 포함하게
    # 된다 - 그래서 임계값을 홀 신호 세기(~4.5mm까지 관측됨)보다 확실히 높고
    # 채널 깊이(~10mm)보다는 낮게 잡아, 홀 자신의 약한 신호는 이 검사에서 아예
    # "깊은 영역"으로 안 잡히고 채널의 강한 신호만 잡히게 한다 (실측으로 확인된
    # 오탐지 수정).
    connection_depth_mm: float = 6.0

    # --- 8방향 섹터 모양(둥근가) 판정 기준 ---
    # 진짜 홀의 residual 자체가 0~4.5mm로 크게 흔들려서, "몇 mm 이상"이라는 절대
    # 크기 기준 대신 "8방향이 서로 고르게 깊은가"(모양)로 판정한다.
    sector_positive_floor_mm: float = 0.3  # 이보다 커야 "그 방향도 파여있다"로 침
    min_sector_positive_fraction: float = 0.6  # 유효 섹터 중 최소 이 비율은 파여있어야 함
    max_sector_variation: float = 1.0  # 변동계수(표준편차/평균) 상한 - 낮을수록 균일(원형)

    # 이 mm 길이 이상인 직선만 "트레드 문양 선"으로 인정한다 - 짧은 노이즈성
    # 선 조각까지 선으로 잡으면, 그 근처의 무관한 진짜 홀까지 덩달아 제외된다.
    min_line_length_mm: float = 12.0

    # 후보 중심이 그 직선에 이 정도(반지름의 배수) 이내로 가까우면 제외한다 -
    # depth 모양 판정과 무관한 독립적 소거 기준. 설계상 홀-채널 간 최소 7mm
    # 클리어런스가 보장돼 있어서, 이 값을 타이트하게 잡아도 진짜 홀은 안 걸린다.
    line_reject_radius_fraction: float = 0.3

    # 트레드 문양 선 검출/제외 기능 자체를 켤지 여부 - 연산 비용 대비 효과를
    # 확인하려고 임시로 끌 수 있게 함. 꺼도 나머지(링/8섹터/그루브 연결성)
    # 검증은 그대로 동작한다.
    enable_line_rejection: bool = False

    # --- 추가 후보 검출 방식 (Hough Circle 두 개에 더해서) ---
    # LoG(Laplacian of Gaussian): 테두리(엣지)가 아니라 "정해진 크기의 둥근
    # 덩어리 자체"를 직접 찾는다 - depression 신호의 테두리가 흐릿해도 덩어리
    # 존재 자체는 잡아낼 수 있어서 Hough가 놓치는 후보를 보완한다.
    # 실측 비교 결과 Hough 단독(끈 상태)이 더 깔끔해서 다시 꺼둔다.
    enable_log_candidates: bool = False

    # 템플릿 매칭: 이번 프레임에서 이미 검증된 홀 하나를 템플릿으로 잘라, 화면
    # 전체에서 그것과 닮은 자리를 추가로 찾는다. Hough/LoG처럼 모양(엣지/블롭)에
    # 기대는 게 아니라 "이미 확인된 진짜 홀과 얼마나 닮았는가"라는 별개 기준.
    # LoG와 같은 이유로 기본은 꺼둔다.
    enable_template_matching: bool = False
    template_match_threshold: float = 0.5  # 정규화 상관계수 최소값

    # --- RGB 그림자/하이라이트 비대칭 검사 ---
    # 조명이 한 방향에서 오면, 진짜 3D 오목한 홀은 한쪽은 그늘지고 반대쪽은
    # 밝아지는 매끈한 방향성 패턴이 생긴다 (평면에 인쇄된 무늬는 이런 패턴이
    # 없다). 8방향 평균 밝기에 코사인 하나를 맞춰서 그 적합도(R^2)를 점수로
    # 쓴다. 아직 실측으로 임계값을 검증하지 않아서 기본은 점수만 진단에 남기고
    # 판정에는 반영하지 않는다 - True로 켜면 min_shadow_highlight_r2 미만은 제외.
    enable_shadow_highlight_check: bool = False
    min_shadow_highlight_r2: float = 0.3

    # --- 스터드(못) 삽입 후 상태 분류 (정상/틀어짐/덜 박힘) ---
    # 홀 탐지(삽입 전)와는 별개 기능. recess_mm 부호 규약(_circle_depth_evidence와
    # 동일): 양수=주변 표면보다 더 멀다(안쪽으로 들어가 있음), 음수=주변보다 더
    # 가깝다(튀어나옴). "정상(=표면과 거의 같은 높이, flush)"은 recess가 0 근처일
    # 때만이고, 양수/음수 어느 쪽으로든 벗어나면 "덜 박힘"이다 - 아직 설계 깊이
    # (3mm)만큼 다 안 들어가서 오목한 굴곡이 남아있어도(recess 양수), 반대로
    # 표면 밖으로 튀어나와도(recess 음수) 둘 다 "제대로 안 박힘"이라는 뜻이라서.
    stud_seating_tolerance_mm: float = 1.0  # |recess|가 이 안이면 표면과 거의 같은 높이(정상)로 봄
    stud_partial_max_depth_mm: float = 3.0  # 설계상 홀 깊이 상한 - 이보다 더 깊게 파인 걸로 나오면
    # (예: 빈 홀을 스터드로 착각) 신뢰할 수 없다고 보고 "원인불명" 처리
    stud_tilt_amplitude_mm: float = 1.2  # 8방향 recess에 맞춘 코사인 진폭이 이보다 커야 "기울어짐" 후보
    stud_tilt_r2: float = 0.6  # 코사인 적합도(R^2)가 이보다 높아야 노이즈가 아니라 진짜 방향성 기울기

    # --- 스터드 머리 위치 탐지 (RGB 색상 기반) ---
    # 타이어 출력물이 무광 검은색이고 스터드(못) 머리는 은색 금속이라 밝기 대비가
    # 뚜렷하다 - depth 굴곡(빈 홀 탐지)과 달리 삽입 전 좌표를 저장해둘 필요 없이,
    # 매 프레임 RGB만으로 스터드 위치를 직접 찾는다. 그래서 카메라나 물체가
    # 움직여도 안전하다.
    stud_head_min_brightness: int = 140  # B/G/R 중 최댓값(0~255) - 이보다 밝아야 금속 후보 (검은 PLA는 훨씬 어두움)
    # HSV 채도(S) 대신 R/G/B 채널 간 최대-최소 편차(chroma)를 직접 본다 - 카드보드
    # 박스/나무 바닥처럼 밝지만 갈색/베이지 색조가 있는 배경이 낮은 HSV 채도로도
    # 통과되는 게 실측으로 확인됨(같은 물체 거리대에 배경이 살짝 걸치는 경우).
    # 진짜 은색 금속은 R≈G≈B라 편차가 훨씬 작다.
    stud_head_max_chroma: int = 25  # R/G/B 최대-최소 편차(0~255) - 이보다 작아야 무채색(은색) 후보


@dataclass
class DetectedHole:
    """[역할] 탐지된 홀 하나의 정보를 담는 결과 객체."""

    pixel: tuple  # 이미지 상의 (u, v) 중심 좌표 (px)
    depth_m: float  # 홀 바닥 depth 추정치, 미터
    hole_depth_mm: float  # 주변 링 대비 홀이 얼마나 깊은지 (mm)
    diameter_px: float
    point_camera_m: np.ndarray = field(repr=False)  # 카메라 좌표계 (x, y, z), 미터


@dataclass
class StudState:
    """[역할] 이미 삽입된 스터드 하나의 상태 판정 결과 (`classify_stud_state` 반환값)."""

    pixel: tuple
    status: str  # "correct" | "tilted" | "under_inserted" | "unknown"
    seating_offset_mm: float | None  # 양수=주변보다 안쪽(들어가있음), 음수=튀어나옴
    tilt_amplitude_mm: float
    tilt_r2: float


def _valid_median(values: np.ndarray) -> float | None:
    """0(무효) 픽셀을 제외한 median. 유효 픽셀이 없으면 None."""
    valid = values[values > 0]
    return float(np.median(valid)) if valid.size else None


class HoleDetector:
    def __init__(self, intrinsics, config: HoleDetectorConfig | None = None):
        self.intrinsics = intrinsics  # 픽셀->3D 변환에 필요한 카메라 내부 파라미터 (rs.intrinsics)
        self.config = config or HoleDetectorConfig()

    def segment_object(self, depth_m: np.ndarray) -> np.ndarray:
        """[역할] 배경을 떼어내고 타이어 출력물만 마스크로 남긴다.

        거리 범위 필터 + 노이즈 제거(열림/닫힘) + 가장 큰 연결 성분 선택.

        닫힘(Closing) 커널은 트레드 채널 폭(최대 5mm, V자/중앙 채널)보다 확실히
        커야 한다 - 그렇지 않으면 채널이 블록들을 서로 다른 "섬"으로 갈라놓고,
        "가장 큰 섬 하나"만 물체로 인정돼서 나머지 블록(과 그 안의 홀)이 전부
        배경으로 취급돼 버린다 (실측으로 확인된 문제). 커널 크기는 가장 가까운
        촬영 거리(worst case, px/mm이 가장 커지는 지점) 기준으로 계산해서
        어떤 거리에서도 채널을 확실히 이어붙이게 한다.
        """
        cfg = self.config
        valid = (depth_m > cfg.min_object_distance_m) & (depth_m < cfg.max_object_distance_m)
        mask = valid.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        worst_case_px_per_mm = self.intrinsics.fx / (cfg.min_object_distance_m * 1000.0)
        closing_px = max(9, int(round(cfg.max_channel_width_mm * worst_case_px_per_mm)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_px, closing_px), np.uint8))

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)

    def _px_per_mm(self, depth_m: np.ndarray, object_mask: np.ndarray) -> float:
        region_depth = depth_m[object_mask > 0]
        median_depth_m = float(np.median(region_depth)) if region_depth.size else 0.3
        return self.intrinsics.fx / (median_depth_m * 1000.0)

    def _make_depression_image(self, depth_mm: np.ndarray, object_mask: np.ndarray, sigma_px: float) -> np.ndarray:
        """[역할] "국소 표면보다 얼마나 더 먼가"를 8비트 밝기로 인코딩한 이미지.

        가우시안 블러로 만든 매끈한 국소 표면(=holes 없다고 가정한 baseline)과
        실측값의 차이를 밝기로 표현한다. 1단위 ≈ 0.5mm, 최대 127.5mm에서 포화.

        sigma는 반드시 명시적으로 지정한다 - GaussianBlur에 sigma=0(자동)을 주면
        커널 크기와 무관하게 실제 블러 반경이 훨씬 작게 계산돼서, 홀 크기에 맞춰
        큰 커널을 줘도 블러가 홀을 제대로 못 지우는 문제가 있었다 (실측으로 확인).
        """
        valid = (depth_mm > 0) & (object_mask > 0)
        if not np.any(valid):
            return np.zeros(depth_mm.shape, dtype=np.uint8)

        fill_value = float(np.median(depth_mm[valid]))
        filled = np.where(valid, depth_mm, fill_value).astype(np.float32)
        filled = cv2.medianBlur(filled, 5)

        # 가우시안 블러는 "공간 거리"만 보고 섞기 때문에, 홀이든 채널(10mm급)이든
        # 가까우면 다 같이 뭉갠다 - 그래서 채널 근처 진짜 홀의 baseline이 오염되는
        # 문제가 있었다. 양방향 필터(bilateral)는 "공간 거리 + 값 차이"를 같이 봐서,
        # sigma_color보다 값 차이가 큰 진짜 경계(채널)는 안 섞고 보존하면서 작은
        # 차이(홀)만 지운다. 설계상 홀-채널 클리어런스가 최소 7mm로 가까울 수
        # 있어서, sigma_color를 너무 크게(예: max_hole_depth_mm 기준) 잡으면
        # 채널 신호가 그 좁은 간격을 넘어 근처 홀의 baseline까지 새어 들어가
        # 홀 신호 자체가 지워지는 문제가 실측+합성 테스트로 확인됐다 - 그래서
        # 홀 신호(수 mm)는 잘 지우되 채널(10mm)과는 확실히 구분되는 낮은
        # 고정값을 쓴다.
        cfg = self.config
        sigma_color_mm = cfg.depression_sigma_color_mm
        diameter_px = min(int(6 * sigma_px + 1) | 1, 61)  # 성능을 위해 상한을 둠
        local_surface = cv2.bilateralFilter(filled, diameter_px, sigma_color_mm, sigma_px)
        depression_mm = np.maximum(filled - local_surface, 0)
        depression_mm[object_mask == 0] = 0

        return np.clip(depression_mm * 2.0, 0, 255).astype(np.uint8)
    """8비트 이미지에서 원처럼 보이는 위치들 찾아 (x,y,r) 후보로 반환"""
    def _hough_candidates(
        self, image: np.ndarray, min_distance_px: float, min_radius_px: int, max_radius_px: int
    ) -> list[tuple[int, int, int]]:
        cfg = self.config
        circles = cv2.HoughCircles(
            cv2.medianBlur(image, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=min_distance_px,
            param1=cfg.hough_param1,
            param2=cfg.hough_param2,
            minRadius=min_radius_px,
            maxRadius=max_radius_px,
        )
        if circles is None:
            return []
        return [tuple(map(int, c)) for c in np.rint(circles[0]).astype(np.int32)]

    @staticmethod
    def _greedy_nms(
        points: np.ndarray, scores: np.ndarray, min_distance_px: float, max_candidates: int = 300
    ) -> list[tuple[int, int]]:
        """[역할] 점수 높은 순으로 정렬해서, 이미 뽑힌 점들과 min_distance_px보다
        가까우면 버리는 탐욕적 비최대억제(greedy NMS).

        `response == maximum_filter(response)` 방식은 평평하거나 값이 같은
        구간(플래토)에서 그 구간 전체가 "극댓값"으로 잡혀버리는 문제가 있었다
        (실측 아닌 균일한 배경에서 후보가 수천 개로 폭증하는 걸로 확인됨) -
        점수 정렬 + 거리 기반 억제는 이 문제가 없다.
        """
        if points.shape[0] == 0:
            return []
        order = np.argsort(-scores)
        if order.size > max_candidates:
            order = order[:max_candidates]  # 성능 보호용 상한
        kept: list[tuple[int, int]] = []
        for index in order:
            x, y = int(points[index, 0]), int(points[index, 1])
            if all((x - kx) ** 2 + (y - ky) ** 2 >= min_distance_px ** 2 for kx, ky in kept):
                kept.append((x, y))
        return kept

    def _log_candidates(
        self, image: np.ndarray, object_mask: np.ndarray, radius_px: int, min_distance_px: float
    ) -> list[tuple[int, int, int]]:
        """[역할] LoG(Laplacian of Gaussian)로 지정한 반지름 크기의 둥근 밝은
        덩어리를 직접 찾는다. Hough Circle은 "테두리(엣지)"가 뚜렷해야 잡히는데,
        depression 신호가 약해서 테두리가 흐릿한 경우에도 "그 자리에 덩어리
        자체가 있다"는 신호는 LoG로 잡힐 수 있어 Hough의 보완책이 된다.
        """
        sigma = max(1.0, radius_px / np.sqrt(2.0))  # 표준 LoG 블롭-반지름 관계
        smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=sigma)
        log = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
        response = -log * (sigma ** 2)  # 스케일 정규화, 밝은 덩어리(=홀) -> 양수 피크
        response[object_mask == 0] = 0

        positive = response[response > 0]
        if positive.size == 0:
            return []
        threshold = max(2.0, float(np.percentile(positive, 90)))
        ys, xs = np.nonzero(response > threshold)
        if xs.size == 0:
            return []
        points = np.column_stack([xs, ys])
        scores = response[ys, xs]
        kept = self._greedy_nms(points, scores, min_distance_px)
        return [(x, y, radius_px) for x, y in kept]

    def _template_candidates(
        self, gray: np.ndarray, reference_holes: list["DetectedHole"], radius_px: int,
        threshold: float, min_distance_px: float,
    ) -> list[tuple[int, int, int]]:
        """[역할] 이미 검증된 진짜 홀 하나를 템플릿으로 잘라, 화면 전체에서 그와
        닮은 자리를 정규화 상관계수(cv2.TM_CCOEFF_NORMED)로 추가 검색한다.
        Hough/LoG가 모양(엣지/블롭)에 기대는 것과 달리 "이미 확인된 진짜 홀과
        얼마나 닮았는가"라는 완전히 다른 기준으로 후보를 낸다.
        """
        if not reference_holes:
            return []
        hx, hy = reference_holes[0].pixel
        r = max(4, int(radius_px))
        x0, y0, x1, y1 = hx - r, hy - r, hx + r, hy + r
        if x0 < 0 or y0 < 0 or x1 >= gray.shape[1] or y1 >= gray.shape[0]:
            return []
        template = gray[y0:y1, x0:x1]
        if template.size == 0 or template.shape[0] < 3 or template.shape[1] < 3:
            return []

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.nonzero(result >= threshold)
        if xs.size == 0:
            return []
        # matchTemplate 결과의 (x,y)는 템플릿의 좌상단 기준이라, 중심 좌표로 되돌리려면
        # 템플릿 반폭(r)만큼 더해야 한다. 상관계수가 높은 진짜 매칭 주변엔 픽셀 수십 개가
        # 다 threshold를 넘기기 때문에(부드러운 상관 곡면), NMS로 하나만 남긴다.
        points = np.column_stack([xs + r, ys + r])
        scores = result[ys, xs]
        kept = self._greedy_nms(points, scores, min_distance_px)
        return [(x, y, radius_px) for x, y in kept]

    def _circle_depth_evidence(self, depth_mm: np.ndarray, object_mask: np.ndarray, x: int, y: int, radius: int):
        """[역할] 후보 원 하나를 감싸는 링(ring) 영역의 depth로 실제 함몰 여부를 검증.

        반환: (recess_mm 또는 None, 중심 무효 비율, 링 유효 비율, 8방향 섹터별
        recess 배열, 링 depth mm). 링 자체가 무효면 None.

        object_mask로 배경(물체 밖)을 제외한다 - 안 그러면 물체 가장자리 근처의
        홀은 링 일부가 훨씬 먼 배경까지 걸쳐서, 그 배경값이 안 걸러지고 섞여
        recess가 수십 mm씩 잘못 계산되는 문제가 있었다 (실측으로 확인됨).
        """
        height, width = depth_mm.shape
        outer_radius = max(int(radius * 1.65), radius + 3)
        left, right = max(0, x - outer_radius), min(width, x + outer_radius + 1)
        top, bottom = max(0, y - outer_radius), min(height, y + outer_radius + 1)
        if right - left < 3 or bottom - top < 3:
            return None

        yy, xx = np.ogrid[top:bottom, left:right]
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        inner = distance2 <= max(2, int(radius * 0.55)) ** 2
        ring = (distance2 >= int(radius * 1.10) ** 2) & (distance2 <= outer_radius ** 2)
        crop = depth_mm[top:bottom, left:right]
        mask_crop = object_mask[top:bottom, left:right] > 0
        inner = inner & mask_crop
        ring = ring & mask_crop

        inner_values, ring_values = crop[inner], crop[ring]
        inner_depth = _valid_median(inner_values)
        ring_depth = _valid_median(ring_values)
        if ring_depth is None:
            return None
        ring_valid_fraction = float(np.mean(ring_values > 0))
        invalid_fraction = float(np.mean(inner_values <= 0))
        recess_mm = None if inner_depth is None else inner_depth - ring_depth

        sector_recesses = []
        angle = np.arctan2(yy - y, xx - x)
        for sector in range(8):
            lower = -np.pi + sector * (np.pi / 4)
            upper = lower + (np.pi / 4)
            sector_ring = ring & (angle >= lower) & (angle < upper)
            sector_depth = _valid_median(crop[sector_ring])
            sector_recesses.append(
                np.nan if inner_depth is None or sector_depth is None else inner_depth - sector_depth
            )
        return recess_mm, invalid_fraction, ring_valid_fraction, np.array(sector_recesses), ring_depth

    def _connects_to_outside_depression(
        self, depression: np.ndarray, x: int, y: int, radius: int, threshold_mm: float
    ) -> bool:
        """[역할] 후보의 "깊은 영역"이 반경 1.75배 밖까지 이어지면 그루브(홈)의
        일부로 보고 True. 진짜 홀은 고립된 국소 함몰이라 이어지지 않는다.
        """
        reach_radius = max(int(np.ceil(radius * 1.75)), radius + 4)
        left, right = max(0, x - reach_radius), min(depression.shape[1], x + reach_radius + 1)
        top, bottom = max(0, y - reach_radius), min(depression.shape[0], y + reach_radius + 1)
        if right - left < 3 or bottom - top < 3:
            return False

        crop = depression[top:bottom, left:right]
        mask = (crop >= int(np.ceil(threshold_mm * 2.0))).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        yy, xx = np.ogrid[top:bottom, left:right]
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        centre = distance2 <= max(2, int(radius * 0.45)) ** 2
        outside = distance2 >= max(int(radius * 1.50), radius + 3) ** 2
        centre_labels = np.unique(labels[centre])
        centre_labels = centre_labels[centre_labels != 0]
        return any(np.any(labels[outside] == label) for label in centre_labels)

    @staticmethod
    def _fit_directional_cosine(values: np.ndarray, angles: np.ndarray) -> tuple[float, float, float]:
        """[역할] 방향(각도)별 값에 코사인 하나(offset + amplitude*cos(angle-phase))를
        최소자승으로 맞춘다. 값이 6개 미만이면 (0,0,0). 반환: (amplitude, r2, phase).

        `_shadow_highlight_r2`(RGB 밝기)와 `classify_stud_state`(depth recess)가
        동일한 적합 로직을 쓰므로 공유 헬퍼로 뺐다.
        """
        if values.size < 6:
            return 0.0, 0.0, 0.0
        design = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        predicted = design @ coeffs
        residual_ss = float(np.sum((values - predicted) ** 2))
        total_ss = float(np.sum((values - np.mean(values)) ** 2))
        r2 = 0.0 if total_ss < 1e-6 else max(0.0, 1.0 - residual_ss / total_ss)
        amplitude = float(np.hypot(coeffs[1], coeffs[2]))
        phase = float(np.arctan2(coeffs[2], coeffs[1]))
        return amplitude, r2, phase

    def _shadow_highlight_r2(self, gray_raw: np.ndarray, x: int, y: int, radius: int) -> float:
        """[역할] 조명이 한 방향에서 올 때 진짜 오목한 홀에 생기는 "한쪽은 그늘,
        반대쪽은 밝음"이라는 매끈한 방향성 패턴을 점수화한다.

        후보를 감싸는 고리를 8방향으로 나눠 평균 밝기를 구하고, 그 8개 값에
        코사인 하나(단일 방향광 모델: 밝기 ≈ a + b·cos(각도-위상))를 최소자승으로
        맞춘 뒤 적합도(R^2, 0~1)를 반환한다. 평면에 인쇄된 무늬는 이런 매끈한
        방향성 그라데이션이 생기지 않아 R^2가 낮게 나오는 경향이 있다.

        CLAHE 보정 이미지가 아니라 원본 밝기(gray_raw)를 써야 한다 - CLAHE는
        타일 단위로 대비를 늘려서, 작은 홀이 타일 경계에 걸치면 인위적인 밝기
        단차가 생겨 그라데이션 측정을 왜곡할 수 있다.
        """
        height, width = gray_raw.shape
        outer_radius = max(int(radius * 1.5), radius + 3)
        left, right = max(0, x - outer_radius), min(width, x + outer_radius + 1)
        top, bottom = max(0, y - outer_radius), min(height, y + outer_radius + 1)
        if right - left < 3 or bottom - top < 3:
            return 0.0

        yy, xx = np.ogrid[top:bottom, left:right]
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        ring = (distance2 >= max(2, int(radius * 0.6)) ** 2) & (distance2 <= outer_radius ** 2)
        angle = np.arctan2(yy - y, xx - x)
        crop = gray_raw[top:bottom, left:right].astype(np.float32)

        means, angles = [], []
        for sector in range(8):
            lower = -np.pi + sector * (np.pi / 4)
            upper = lower + np.pi / 4
            values = crop[ring & (angle >= lower) & (angle < upper)]
            if values.size:
                means.append(float(np.mean(values)))
                angles.append(lower + np.pi / 8)
        _amplitude, r2, _phase = self._fit_directional_cosine(np.array(means), np.array(angles))
        return r2

    def _detect_lines(self, gray: np.ndarray, px_per_mm: float) -> np.ndarray | None:
        """[역할] RGB(CLAHE 보정 흑백)에서 직선(트레드 문양 선) 구간을 찾는다.

        진짜 홀은 평평한 블록 중앙에 있어서 이런 뚜렷한 직선 위에 있을 수 없다 -
        depth 모양(원형도) 판정과 무관한, 완전히 독립적인 소거 기준이라 depth
        판정이 애매하게 통과시킨 채널 교차점 등을 추가로 걸러낼 수 있다.

        최소 길이를 실제 mm 기준으로 잡는다 - 짧은 노이즈성 선 조각까지 "트레드
        문양 선"으로 오인하면, 그 근처의 무관한 진짜 홀까지 덩달아 제외돼버린다.
        트레드 문양 선은 리브를 가로지르는 긴 선이라 최소 길이를 충분히 크게
        잡아도 실제 문양 선은 여전히 잡힌다.
        """
        min_length_px = max(20, int(round(self.config.min_line_length_mm * px_per_mm)))
        edges = cv2.Canny(gray, 50, 150)
        return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_length_px, maxLineGap=5)

    @staticmethod
    def _distance_to_nearest_line(lines: np.ndarray | None, x: float, y: float) -> float:
        if lines is None:
            return float("inf")
        best = float("inf")
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).ravel()[:4]
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                distance = float(np.hypot(x - x1, y - y1))
            else:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
                proj_x, proj_y = x1 + t * dx, y1 + t * dy
                distance = float(np.hypot(x - proj_x, y - proj_y))
            best = min(best, distance)
        return best

    def detect(self, color_bgr: np.ndarray, depth_m: np.ndarray):
        """[역할] 메인 파이프라인: 물체 분리 -> Hough로 원형 후보 검출(RGB+depth
        양쪽) -> 후보별 링/8섹터/연결성 검증 -> 3D 좌표 계산. (holes, debug) 반환.
        """
        cfg = self.config
        object_mask = self.segment_object(depth_m)
        if cv2.countNonZero(object_mask) == 0:
            empty = np.zeros_like(object_mask)
            return [], {"object_mask": object_mask, "depression": empty, "candidate_count": 0}

        px_per_mm = self._px_per_mm(depth_m, object_mask)
        depth_mm = depth_m * 1000.0

        min_radius_px = max(2, int(round(cfg.min_hole_diameter_mm / 2 * px_per_mm)))
        max_radius_px = max(min_radius_px + 1, int(round(cfg.max_hole_diameter_mm / 2 * px_per_mm)))
        min_distance_px = max(4.0, cfg.min_candidate_distance_mm * px_per_mm)
        sigma_px = cfg.surface_blur_sigma_px or (max_radius_px * 2.5)

        depression = self._make_depression_image(depth_mm, object_mask, sigma_px)
        default_radius_px = (min_radius_px + max_radius_px) // 2

        gray_raw = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)  # 그림자/하이라이트 점수용 (CLAHE 전)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_raw)
        gray = np.where(object_mask > 0, gray, 0).astype(np.uint8)
        # 트레드 문양 선 검출 - 연산 비용 대비 효과 비교를 위해 임시로 끌 수 있음
        lines = self._detect_lines(gray, px_per_mm) if cfg.enable_line_rejection else None

        candidates = self._hough_candidates(depression, min_distance_px, min_radius_px, max_radius_px)
        candidates += self._hough_candidates(gray, min_distance_px, min_radius_px, max_radius_px)
        if cfg.enable_log_candidates:
            # LoG는 Hough와 달리 테두리가 아니라 "그 크기의 둥근 덩어리 자체"를
            # 찾으므로 단일 대표 반지름(min~max 중간값)으로 한 번만 돈다.
            candidates += self._log_candidates(depression, object_mask, default_radius_px, min_distance_px)

        unique_candidates: list[tuple[int, int, int]] = []
        for cx, cy, cr in candidates:
            if all((cx - ux) ** 2 + (cy - uy) ** 2 > (min(cr, ur) * 0.7) ** 2 for ux, uy, ur in unique_candidates):
                unique_candidates.append((cx, cy, cr))

        object_bbox = cv2.boundingRect(object_mask)  # (x, y, w, h)
        holes = self._verify_candidates(unique_candidates, depth_mm, depression, object_mask, object_bbox, lines, gray_raw)

        # 템플릿 매칭 2차 패스: 1차에서 홀이 하나라도 확정됐으면, 그걸 템플릿으로
        # 잘라서 화면 전체에서 닮은 자리를 추가로 찾는다 (Hough/LoG가 놓친 후보 보완).
        if cfg.enable_template_matching and holes:
            template_candidates = self._template_candidates(
                gray, holes, default_radius_px, cfg.template_match_threshold, min_distance_px
            )
            # 중복 판단 기준은 "같은 후보 원끼리 겹치는지"(반지름 기준, 위 1차 dedup)가
            # 아니라 "이미 확정된 홀과 같은 물리적 구멍인지"(실제 홀 간격 기준)라서
            # min_distance_px를 써야 한다 - 반지름의 0.7배(~11~14px)는 너무 좁아서
            # 상관관계 곡면이 부드러운 템플릿 매칭에서 같은 홀 주변 몇 픽셀 떨어진
            # 점도 "새 후보"로 통과해 중복 검출되는 버그가 있었다 (실측으로 확인).
            existing_xy = [(h.pixel[0], h.pixel[1]) for h in holes]
            new_unique: list[tuple[int, int, int]] = []
            for cx, cy, cr in template_candidates:
                too_close_to_existing = any(
                    (cx - ux) ** 2 + (cy - uy) ** 2 < min_distance_px ** 2 for ux, uy in existing_xy
                )
                too_close_to_new = any(
                    (cx - nx) ** 2 + (cy - ny) ** 2 < min_distance_px ** 2 for nx, ny, _ in new_unique
                )
                if not too_close_to_existing and not too_close_to_new:
                    new_unique.append((cx, cy, cr))
            if new_unique:
                extra_holes = self._verify_candidates(new_unique, depth_mm, depression, object_mask, object_bbox, lines, gray_raw)
                holes = holes + extra_holes
                unique_candidates = unique_candidates + new_unique

        debug = {
            "object_mask": object_mask,
            "depression": depression,
            "candidate_count": len(unique_candidates),
            "px_per_mm": px_per_mm,
            # 진단 도구(model.py 클릭 콜백)가 임의의 픽셀에 대해 왜 통과/탈락하는지
            # 재계산 없이 바로 확인할 수 있도록 중간 산출물을 그대로 넘겨준다.
            "depth_mm": depth_mm,
            "lines": lines,
            "object_bbox": object_bbox,
            "default_radius_px": default_radius_px,
            "gray_raw": gray_raw,
        }
        return holes, debug

    def _verify_candidates(self, candidates, depth_mm, depression, object_mask, object_bbox, lines, gray_raw):
        holes: list[DetectedHole] = []
        for x, y, radius in candidates:
            hole, _diagnostics = self._evaluate_candidate(
                x, y, radius, depth_mm, depression, object_mask, object_bbox, lines, gray_raw
            )
            if hole is not None:
                holes.append(hole)
        return holes

    def diagnose(
        self, x: int, y: int, depth_mm: np.ndarray, depression: np.ndarray,
        object_mask: np.ndarray, object_bbox: tuple, lines, gray_raw: np.ndarray | None = None,
        radius: int | None = None,
    ) -> dict:
        """[역할] 임의의 픽셀 하나가 왜 홀로 인정되는지/안 되는지 진단 정보를 반환한다
        (model.py 클릭 디버그 도구용). `_verify_candidates`와 완전히 같은 판정
        로직을 공유하므로(`_evaluate_candidate`), 실제 파이프라인과 다른 결과가
        나올 걱정 없이 "이 자리는 왜 탈락했나"를 정확히 확인할 수 있다.
        """
        if radius is None:
            radius = max(3, int(round((self.config.min_hole_diameter_mm + self.config.max_hole_diameter_mm)
                                       / 4 * self._px_per_mm_from_depth(depth_mm, object_mask))))
        if gray_raw is None:
            gray_raw = np.zeros(depth_mm.shape, dtype=np.uint8)
        _hole, diagnostics = self._evaluate_candidate(
            x, y, radius, depth_mm, depression, object_mask, object_bbox, lines, gray_raw
        )
        return diagnostics

    def _px_per_mm_from_depth(self, depth_mm: np.ndarray, object_mask: np.ndarray) -> float:
        region = depth_mm[object_mask > 0]
        median_depth_mm = float(np.median(region)) if region.size else 300.0
        return self.intrinsics.fx / median_depth_mm

    def _face_sector_depths(
        self, depth_mm: np.ndarray, object_mask: np.ndarray, x: int, y: int, radius: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """[역할] 스터드 머리 표면 자체를 8방향으로 나눠 각 방향의 depth 중앙값을 반환.

        `_circle_depth_evidence`의 sector_recesses는 "주변 링이 방향별로 얼마나
        고르게 깊은가"(그루브 판별용)라서, 스터드 머리 표면 안쪽의 기울기는
        전혀 못 잡는다(내부는 중심 하나의 중앙값으로만 뭉뚱그려짐). 기울어짐은
        머리 표면 자체 안에서 한쪽은 더 들어가고 반대쪽은 덜 들어간 경사이므로,
        여기서는 반대로 머리 표면(중심 근처는 제외 - 굴곡/노이즈가 큼) 안쪽을
        8방향으로 나눠 절대 depth 값 자체의 방향성을 본다.
        """
        height, width = depth_mm.shape
        face_outer = max(2, int(round(radius * 0.85)))
        face_inner = max(1, int(round(radius * 0.25)))
        left, right = max(0, x - face_outer), min(width, x + face_outer + 1)
        top, bottom = max(0, y - face_outer), min(height, y + face_outer + 1)
        if right - left < 3 or bottom - top < 3:
            return None

        yy, xx = np.ogrid[top:bottom, left:right]
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        face = (distance2 >= face_inner ** 2) & (distance2 <= face_outer ** 2)
        face = face & (object_mask[top:bottom, left:right] > 0)
        crop = depth_mm[top:bottom, left:right]
        angle = np.arctan2(yy - y, xx - x)

        depths, angles = [], []
        for sector in range(8):
            lower = -np.pi + sector * (np.pi / 4)
            upper = lower + np.pi / 4
            sector_depth = _valid_median(crop[face & (angle >= lower) & (angle < upper)])
            if sector_depth is not None:
                depths.append(sector_depth)
                angles.append(lower + np.pi / 8)
        if len(depths) < 6:
            return None
        return np.array(depths), np.array(angles)

    def classify_stud_state(
        self, depth_mm: np.ndarray, object_mask: np.ndarray, x: int, y: int, radius: int
    ) -> StudState:
        """[역할] 이미 스터드가 박힌 자리(삽입 전 홀 탐지로 얻은 x,y,radius를 그대로
        재사용한다고 가정 - 카메라/물체가 삽입 도중 움직이지 않아야 함)에서, 그
        상태를 "correct"(정확히 박힘) / "tilted"(틀어짐) / "under_inserted"(덜 박힘)
        중 하나로 판정한다. 홀 탐지와 판정 기준이 다르므로 accept/reject가 아니라
        독립적인 분류 함수다.

        - 전체적으로 얼마나 안쪽/바깥쪽에 있는지(seating_offset_mm)는
          `_circle_depth_evidence`의 recess_mm(머리 중심 vs 주변 표면 링)을
          그대로 재사용한다 - 양수=주변보다 안쪽/멀다(아직 설계 깊이만큼 다 안
          들어가 오목한 굴곡이 남음), 음수=튀어나옴/가깝다(표면 밖으로 나옴).
          "정상(flush)"은 이 값이 0 근처(허용 오차 이내)일 때뿐이고, 양쪽 어느
          방향으로 벗어나도 "덜 박힘"이다. recess가 설계 홀 깊이 상한
          (`stud_partial_max_depth_mm`)보다도 크면 신뢰할 수 없는 값(예: 빈
          홀을 스터드로 착각)으로 보고 "원인불명" 처리한다.
        - 기울어짐은 `_face_sector_depths`로 머리 표면 자체의 방향별 depth를
          구해 코사인 하나를 맞춘 뒤(`_fit_directional_cosine`) 진폭과 적합도
          (R^2)를 본다 - 진짜 기울기는 둘 다 기준 이상, 노이즈는 진폭이 있어도
          R^2가 낮게 나온다.
        """
        cfg = self.config
        evidence = self._circle_depth_evidence(depth_mm, object_mask, x, y, radius)
        if evidence is None:
            return StudState((x, y), "unknown", None, 0.0, 0.0)
        recess_mm, _invalid_fraction, ring_valid_fraction, _sector_recesses, _ring_depth_mm = evidence
        if recess_mm is None or ring_valid_fraction < 0.75:
            return StudState((x, y), "unknown", recess_mm, 0.0, 0.0)
        if recess_mm > cfg.stud_partial_max_depth_mm:
            return StudState((x, y), "unknown", recess_mm, 0.0, 0.0)

        face_data = self._face_sector_depths(depth_mm, object_mask, x, y, radius)
        if face_data is None:
            return StudState((x, y), "unknown", recess_mm, 0.0, 0.0)
        face_depths, face_angles = face_data
        amplitude, r2, _phase = self._fit_directional_cosine(face_depths, face_angles)

        if amplitude >= cfg.stud_tilt_amplitude_mm and r2 >= cfg.stud_tilt_r2:
            status = "tilted"
        elif abs(recess_mm) > cfg.stud_seating_tolerance_mm:
            status = "under_inserted"
        else:
            status = "correct"

        return StudState((x, y), status, recess_mm, amplitude, r2)

    def _silver_mask(self, color_bgr: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """[역할] 검은 타이어 위에서 은색(금속) 픽셀만 True인 불리언 마스크.

        무광 검은 PLA는 거의 항상 어둡고, 금속 스터드 머리는 밝고 무채색(R≈G≈B)
        이라 대비가 뚜렷하다. "무채색"은 HSV 채도가 아니라 R/G/B 채널 간
        최대-최소 편차(chroma)로 직접 판단한다 - 카드보드 박스/나무 바닥처럼
        밝지만 갈색/베이지 색조가 있는 배경이 HSV 채도만으로는 걸러지지 않고
        후보로 잡히는 게 실측으로 확인됐다(물체와 비슷한 거리에 배경이 살짝
        걸쳐 object_mask 안에 들어온 경우). 진짜 은색 금속은 R/G/B가 서로
        거의 같아서 chroma가 훨씬 작다.
        """
        cfg = self.config
        color_i16 = color_bgr.astype(np.int16)
        channel_max = color_i16.max(axis=2)
        chroma = channel_max - color_i16.min(axis=2)
        return (
            (channel_max >= cfg.stud_head_min_brightness)
            & (chroma <= cfg.stud_head_max_chroma)
            & (object_mask > 0)
        )

    def _detect_stud_head_candidates(
        self, color_bgr: np.ndarray, object_mask: np.ndarray, px_per_mm: float
    ) -> list[tuple[int, int, int]]:
        """[역할] 은색 픽셀이 뭉친 덩어리를 찾아 (x,y,radius) 후보로 반환한다.

        완전히 박혀 표면과 거의 같은 높이(flush)라 depth 굴곡이 거의 없는
        스터드도 이걸로 잡힌다 - depth 굴곡(빈 홀 탐지용)과 달리 이전 프레임
        좌표를 기억할 필요가 없다. (아직 설계 깊이만큼 안 들어가 굴곡이 남아
        있는 "덜 박힘" 케이스는 `detect_stud_states`가 별도 경로로 찾는다 -
        가려진 부분이 많아 여기 면적/밝기 기준을 못 넘기는 경우가 있어서.)
        """
        cfg = self.config
        mask = self._silver_mask(color_bgr, object_mask).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        min_radius_px = max(2, int(round(cfg.min_hole_diameter_mm / 2 * px_per_mm)))
        max_radius_px = max(min_radius_px + 1, int(round(cfg.max_hole_diameter_mm / 2 * px_per_mm)))
        # 반사 얼룩 등으로 완벽한 원이 아닐 수 있어 면적 기준에 여유를 둔다.
        min_area = np.pi * (min_radius_px * 0.6) ** 2
        max_area = np.pi * (max_radius_px * 1.3) ** 2

        num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for label in range(1, num):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[label]
            radius_px = max(2, int(round(np.sqrt(area / np.pi))))
            candidates.append((int(round(cx)), int(round(cy)), radius_px))
        return candidates

    def detect_stud_states(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[list[StudState], dict]:
        """[역할] 이미 삽입된 스터드를 찾아 각각의 상태를 판정한다 (holes와
        대응되는 메인 진입점). 삽입 전 좌표를 저장해둘 필요가 없다 - 매 프레임
        그 자리에서 새로 찾기 때문에 카메라/물체가 움직여도 안전하다.

        두 경로로 후보 위치를 찾아 합친다:
        1. RGB 은색 블롭(`_detect_stud_head_candidates`) - 완전히 박혀 표면과
           거의 같은 높이(flush)라 depth 굴곡이 거의 없는 스터드를 잡는다.
        2. 빈 홀과 같은 depth 굴곡 후보(Hough) 중, 그 안에 은색 픽셀이 하나라도
           있는 것만 - 아직 설계 깊이만큼 안 들어가 굴곡이 남아있으면서(그래서
           빈 홀 탐지 방식으로는 잡히지만) 안쪽 그늘 때문에 보이는 금속 면적이
           작아 경로 1의 면적/밝기 기준을 못 넘기는 "덜 박힘" 케이스를 잡는다.
           은색이 전혀 없으면 그냥 진짜 빈 홀이므로 후보에서 뺀다.
        """
        cfg = self.config
        object_mask = self.segment_object(depth_m)
        if cv2.countNonZero(object_mask) == 0:
            return [], {"object_mask": object_mask}

        px_per_mm = self._px_per_mm(depth_m, object_mask)
        depth_mm = depth_m * 1000.0
        min_radius_px = max(2, int(round(cfg.min_hole_diameter_mm / 2 * px_per_mm)))
        max_radius_px = max(min_radius_px + 1, int(round(cfg.max_hole_diameter_mm / 2 * px_per_mm)))
        min_distance_px = max(4.0, cfg.min_candidate_distance_mm * px_per_mm)

        candidates = list(self._detect_stud_head_candidates(color_bgr, object_mask, px_per_mm))

        sigma_px = cfg.surface_blur_sigma_px or (max_radius_px * 2.5)
        depression = self._make_depression_image(depth_mm, object_mask, sigma_px)
        depth_candidates = self._hough_candidates(depression, min_distance_px, min_radius_px, max_radius_px)
        silver_mask = self._silver_mask(color_bgr, object_mask)
        height, width = silver_mask.shape
        for x, y, r in depth_candidates:
            top, bottom = max(0, y - r), min(height, y + r + 1)
            left, right = max(0, x - r), min(width, x + r + 1)
            if not np.any(silver_mask[top:bottom, left:right]):
                continue  # 은색이 전혀 없음 - 진짜 빈 홀, 스터드 후보 아님
            if all((x - ux) ** 2 + (y - uy) ** 2 >= min_distance_px ** 2 for ux, uy, _ur in candidates):
                candidates.append((x, y, r))

        states = [self.classify_stud_state(depth_mm, object_mask, x, y, r) for x, y, r in candidates]
        return states, {"object_mask": object_mask}

    def _evaluate_candidate(self, x, y, radius, depth_mm, depression, object_mask, object_bbox, lines, gray_raw):
        """[역할] 후보 하나를 검증해 (DetectedHole 또는 None, 진단 dict)를 반환한다.

        진단 dict의 "reject_reason"에 어느 단계에서 왜 떨어졌는지 문자열로 남긴다 -
        `_verify_candidates`(전체 파이프라인)와 `diagnose`(클릭 디버그) 양쪽에서
        동일한 로직을 그대로 재사용한다.
        """
        cfg = self.config
        height, width = depth_mm.shape
        diag: dict = {"pixel": (x, y), "radius_px": radius}

        if not (0 <= x < width and 0 <= y < height) or object_mask[y, x] == 0:
            diag["reject_reason"] = "물체(object_mask) 밖"
            return None, diag

        line_distance = self._distance_to_nearest_line(lines, x, y)
        diag["line_distance_px"] = line_distance
        if line_distance < radius * cfg.line_reject_radius_fraction:
            diag["reject_reason"] = f"트레드 문양 선에 너무 가까움 (거리 {line_distance:.1f}px)"
            return None, diag

        evidence = self._circle_depth_evidence(depth_mm, object_mask, x, y, radius)
        if evidence is None:
            diag["reject_reason"] = "주변 링(ring) depth가 전부 무효"
            return None, diag
        recess_mm, invalid_fraction, ring_valid_fraction, sector_recesses, ring_depth_mm = evidence
        diag.update(recess_mm=recess_mm, invalid_fraction=invalid_fraction, ring_valid_fraction=ring_valid_fraction)

        diameter_mm = (2.0 * radius * ring_depth_mm) / self.intrinsics.fx
        diag["diameter_mm"] = diameter_mm
        if diameter_mm < cfg.min_hole_diameter_mm or diameter_mm > cfg.max_hole_diameter_mm * 1.5:
            diag["reject_reason"] = f"환산 지름 {diameter_mm:.1f}mm이 허용 범위 밖"
            return None, diag

        if self._connects_to_outside_depression(depression, x, y, radius, cfg.connection_depth_mm):
            diag["reject_reason"] = "깊은 영역이 반경 밖까지 이어짐 (그루브로 판단)"
            return None, diag

        # 가장자리(물체 테두리 근처) 완화 없이, 어디서든 동일하게 엄격한 기준을
        # 적용한다 - 완화 구간이 하단 등 특정 위치의 오탐지를 늘리는 원인이었다.
        required_ring_valid = 0.75

        if ring_valid_fraction < required_ring_valid:
            diag["reject_reason"] = f"링 유효 비율 {ring_valid_fraction:.2f} < 기준 {required_ring_valid}"
            return None, diag

        # 실측 결과 진짜 홀의 residual 자체가 0~4.5mm로 5배 넘게 흔들려서, "몇 mm
        # 이상 깊어야 한다"는 절대 크기 기준은 근본적으로 불안정하다. 대신 "모양이
        # 둥근가"(=8방향이 서로 고르게 깊은가)로 판단을 바꾼다 - 홈(채널)은 지나가는
        # 1~2방향만 깊고 나머지는 거의 0이라 편차가 크고, 진짜 원형 홀은 신호
        # 세기와 무관하게 사방에서 고르게 깊다.
        valid_sectors = sector_recesses[~np.isnan(sector_recesses)]
        required_valid_sectors = 6
        diag["valid_sector_count"] = int(valid_sectors.size)
        if valid_sectors.size < required_valid_sectors:
            diag["reject_reason"] = f"유효 섹터 {valid_sectors.size}개 < 기준 {required_valid_sectors}"
            return None, diag

        # 평균/표준편차 대신 중앙값(median)/MAD(중앙값 기준 절대편차)를 쓴다 -
        # 근처 채널 때문에 8방향 중 1~2방향만 오염되는 경우(실측으로 확인됨)에도
        # 평균은 소수의 오염된 방향에 쉽게 끌려가지만, 중앙값은 절반 미만이 오염된
        # 이상 영향을 안 받는다.
        median_recess = float(np.median(valid_sectors))
        positive_fraction = float(np.mean(valid_sectors > cfg.sector_positive_floor_mm))
        mad = float(np.median(np.abs(valid_sectors - median_recess)))
        uniformity_cv = mad / (abs(median_recess) + cfg.sector_positive_floor_mm)
        diag.update(median_recess_mm=median_recess, positive_fraction=positive_fraction, uniformity_cv=uniformity_cv)

        required_positive_fraction = cfg.min_sector_positive_fraction
        max_uniformity_cv = cfg.max_sector_variation
        diag.update(required_positive_fraction=required_positive_fraction, max_uniformity_cv=max_uniformity_cv)

        accepted = (
            recess_mm is not None
            and cfg.min_hole_depth_mm <= median_recess <= cfg.max_hole_depth_mm
            and positive_fraction >= required_positive_fraction
            and uniformity_cv <= max_uniformity_cv
        )
        # 아주 좁고 깊은 홀은 중심부 depth 측정 자체가 실패(무효)할 수 있다 -
        # 그 경우 링이 충분히 유효하고 중심 무효 비율이 높으면 홀로 인정.
        if not accepted and recess_mm is None:
            accepted = invalid_fraction >= 0.65 and ring_valid_fraction >= 0.95

        if not accepted:
            if recess_mm is None:
                diag["reject_reason"] = "중심 depth 무효, 대체 조건도 불충족"
            elif not (cfg.min_hole_depth_mm <= median_recess <= cfg.max_hole_depth_mm):
                diag["reject_reason"] = f"중앙값 recess {median_recess:.2f}mm가 [{cfg.min_hole_depth_mm}, {cfg.max_hole_depth_mm}] 밖"
            elif positive_fraction < required_positive_fraction:
                diag["reject_reason"] = f"양의 방향 비율 {positive_fraction:.2f} < 기준 {required_positive_fraction}"
            else:
                diag["reject_reason"] = f"방향별 변동계수 {uniformity_cv:.2f} > 기준 {max_uniformity_cv:.2f} (모양이 안 둥긂)"
            return None, diag

        # RGB 그림자/하이라이트 점수 - depth 검증을 통과한 후보에 한해 계산한다
        # (모든 후보마다 계산하면 낭비이므로). 기본은 진단 정보로만 남기고,
        # enable_shadow_highlight_check가 켜져 있으면 실제 판정에도 반영한다.
        shadow_highlight_r2 = self._shadow_highlight_r2(gray_raw, x, y, radius)
        diag["shadow_highlight_r2"] = shadow_highlight_r2
        if cfg.enable_shadow_highlight_check and shadow_highlight_r2 < cfg.min_shadow_highlight_r2:
            diag["reject_reason"] = (
                f"그림자/하이라이트 패턴 적합도 {shadow_highlight_r2:.2f} < 기준 {cfg.min_shadow_highlight_r2} "
                "(평면 무늬로 의심)"
            )
            return None, diag

        diag["reject_reason"] = None  # 통과
        # 표시/반환하는 깊이는 판정에 실제로 쓰인 median_recess를 쓴다 - 중심점만
        # 보는 raw recess_mm을 썼더니, 통과 판정(중앙값 기준)과 표시값(중심점 기준)이
        # 서로 달라 "홀로 인정됐는데 깊이가 음수"처럼 모순된 값이 나온 적이 있었다.
        reported_depth_mm = median_recess if recess_mm is not None else cfg.min_hole_depth_mm
        floor_depth_mm = ring_depth_mm + reported_depth_mm
        point = rs_deproject(self.intrinsics, x, y, floor_depth_mm / 1000.0)
        hole = DetectedHole(
            pixel=(x, y),
            depth_m=floor_depth_mm / 1000.0,
            hole_depth_mm=reported_depth_mm,
            diameter_px=2.0 * radius,
            point_camera_m=point,
        )
        return hole, diag


def rs_deproject(intrinsics, u: float, v: float, depth_m: float) -> np.ndarray:
    """[역할] 이미지 픽셀(u, v) + depth 값을 카메라 좌표계 3D 점 (x, y, z)로 변환."""
    import pyrealsense2 as rs
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m))
    return np.array(point, dtype=float)


class HoleTracker:
    """[역할] 프레임마다 다시 계산되는 홀 목록을, 화면에서 깜빡이지 않고 계속
    타겟팅하는 것처럼 보이도록 안정화한다 (`depth_v1.py`의 HoleTracker 참고).

    매 프레임 독립적으로 탐지하면 노이즈 때문에 원이 나타났다 사라졌다 하는데,
    이 트래커는 "가까운 위치에 여러 프레임 연속으로 잡혀야" 화면에 내보내고,
    잠깐 놓쳐도(`miss_tolerance`까지) 바로 사라지지 않게 유지한다.
    """

    def __init__(self, confirm_frames: int = 3, miss_tolerance: int = 10):
        self.confirm_frames = confirm_frames
        self.miss_tolerance = miss_tolerance
        self._tracks: list[dict] = []

    def clear(self) -> None:
        self._tracks.clear()

    def update(self, detections: list[DetectedHole]) -> list[DetectedHole]:
        matched: set[int] = set()
        for hole in detections:
            x, y = hole.pixel
            radius = hole.diameter_px / 2.0
            best_index, best_distance = None, float("inf")
            for index, track in enumerate(self._tracks):
                if index in matched:
                    continue
                distance = float(np.hypot(x - track["x"], y - track["y"]))
                # Hough 중심이 프레임마다 몇 px씩 흔들릴 수 있어서 표시 크기보다 넉넉하게 잡음
                limit = max(14.0, max(radius, track["radius"]) * 1.3)
                if distance < limit and distance < best_distance:
                    best_index, best_distance = index, distance

            if best_index is None:
                self._tracks.append({
                    "x": float(x), "y": float(y), "radius": radius,
                    "hole": hole, "streak": 1, "misses": 0, "confirmed": False,
                })
                matched.add(len(self._tracks) - 1)
                continue

            track = self._tracks[best_index]
            track["x"] = 0.75 * track["x"] + 0.25 * x
            track["y"] = 0.75 * track["y"] + 0.25 * y
            track["radius"] = 0.75 * track["radius"] + 0.25 * radius
            track["hole"] = hole
            track["streak"] += 1  # 연속 적중 횟수 - 한 번이라도 놓치면 0으로 끊김
            track["misses"] = 0
            if track["streak"] >= self.confirm_frames:
                track["confirmed"] = True  # 한 번 확정되면 이후 잠깐 놓쳐도 유지
            matched.add(best_index)

        for index, track in enumerate(self._tracks):
            if index not in matched:
                track["misses"] += 1
                track["streak"] = 0  # 놓친 순간 연속 기록 초기화 - 누적되던 버그 수정
        self._tracks = [t for t in self._tracks if t["misses"] <= self.miss_tolerance]

        # "confirmed"는 confirm_frames번 연속 적중해야만 True가 된다 (버그 수정 전에는
        # 놓쳐도 안 끊기는 누적 카운터였어서, 시간이 지날수록 잡음이 어쩌다 쌓여
        # 오탐지가 계속 늘어났다).
        visible = [t for t in self._tracks if t["confirmed"]]
        result = []
        for track in visible:
            hole = track["hole"]
            result.append(DetectedHole(
                pixel=(int(round(track["x"])), int(round(track["y"]))),
                depth_m=hole.depth_m,
                hole_depth_mm=hole.hole_depth_mm,
                diameter_px=track["radius"] * 2.0,
                point_camera_m=hole.point_camera_m,
            ))
        return result
