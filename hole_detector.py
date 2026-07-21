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
3. 후보 주변을 8방향으로 나눠 "모든 방향에서 깊은지"를 확인한다 - 진짜 둥근 홀은
   사방에서 깊고, 트레드 홈(V자 띠)은 홈이 지나가는 한두 방향에서만 깊으므로 이
   차이로 확실하게 구분한다.
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

    # 물체(타이어 출력물) 분리용 거리 범위 - 카메라를 타이어 정면 20~40cm로
    # 옮긴 구도 기준.
    min_object_distance_m: float = 0.18
    max_object_distance_m: float = 0.45

    # 트레드에서 가장 넓은 채널의 폭(mm) - tire_chevron_v4.stl 기준 V자/중앙
    # 채널이 5mm로 가장 넓음. segment_object의 Closing 커널이 이보다 확실히
    # 커야 채널 때문에 블록들이 서로 다른 섬으로 갈라지지 않는다.
    max_channel_width_mm: float = 6.0

    # 핀 머리가 들어가는 입구 지름 11mm 기준, 프린팅 오차 대비 여유
    min_hole_diameter_mm: float = 9.0
    max_hole_diameter_mm: float = 13.0

    # 후보 중심이 주변 링보다 몇 mm 더 깊어야(=멀어야) "진짜 홀"로 인정할지
    min_hole_depth_mm: float = 0.8
    max_hole_depth_mm: float = 2.5

    # Hough Circle 파라미터 (Canny 상위 임계값 / 누산기 임계값 - 낮을수록 후보가 늘어남).
    # 우리 depression 이미지는 실측 신호가 약해서(1~2mm ≈ 밝기 2~4단계) 대비가 매우
    # 낮다 - 원본(depth_v1.py 기본값 50/16)은 이 신호 세기에서 후보를 아예 못 찾아서
    # 낮췄다. 후보가 늘어나는 대신 링/8섹터/연결성 검증 단계가 걸러낸다.
    hough_param1: float = 10.0
    hough_param2: float = 5.0

    # 후보 원 중심 사이 최소 간격(mm) - px는 매 프레임 거리로 자동 환산
    min_candidate_distance_mm: float = 15.0

    # 국소 표면(baseline) 추정용 가우시안 블러의 sigma(px). 0이면 최대 홀 반지름의
    # 약 2.5배로 매 프레임 자동 계산 - 실측 검증 결과 sigma가 홀 반지름의 최소
    # 2~2.5배는 돼야 블러가 홀을 확실히 "지우고" 진짜 배경 표면을 복원한다
    # (sigma가 반지름과 비슷한 정도면 홀 자신의 값이 local_surface에 새어 들어가
    # residual이 실제보다 훨씬 작게 나옴).
    surface_blur_sigma_px: float = 0.0

    # 후보의 깊은 영역이 이 mm 임계값 기준으로 반경 밖까지 이어지면 그루브(홈)의
    # 일부로 보고 제외한다.
    connection_depth_mm: float = 1.0

    # 8방향 섹터 중 몇 개 이상에서 깊어야 "둥근 홀"로 인정할지 (화면 중앙 기준,
    # 가장자리는 스테레오 매칭 품질이 떨어져서 자동으로 완화됨)
    required_deep_sectors: int = 5


@dataclass
class DetectedHole:
    """[역할] 탐지된 홀 하나의 정보를 담는 결과 객체."""

    pixel: tuple  # 이미지 상의 (u, v) 중심 좌표 (px)
    depth_m: float  # 홀 바닥 depth 추정치, 미터
    hole_depth_mm: float  # 주변 링 대비 홀이 얼마나 깊은지 (mm)
    diameter_px: float
    point_camera_m: np.ndarray = field(repr=False)  # 카메라 좌표계 (x, y, z), 미터


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

        kernel_px = int(6 * sigma_px + 1) | 1  # OpenCV 권장: 커널 >= 6*sigma+1
        local_surface = cv2.GaussianBlur(filled, (kernel_px, kernel_px), sigma_px)
        depression_mm = np.maximum(filled - local_surface, 0)
        depression_mm[object_mask == 0] = 0

        return np.clip(depression_mm * 2.0, 0, 255).astype(np.uint8)

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

    def _circle_depth_evidence(self, depth_mm: np.ndarray, x: int, y: int, radius: int):
        """[역할] 후보 원 하나를 감싸는 링(ring) 영역의 depth로 실제 함몰 여부를 검증.

        반환: (recess_mm 또는 None, 중심 무효 비율, 링 유효 비율, 8방향 섹터별
        recess 배열, 링 depth mm). 링 자체가 무효면 None.
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

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = np.where(object_mask > 0, gray, 0).astype(np.uint8)

        candidates = self._hough_candidates(depression, min_distance_px, min_radius_px, max_radius_px)
        candidates += self._hough_candidates(gray, min_distance_px, min_radius_px, max_radius_px)

        unique_candidates: list[tuple[int, int, int]] = []
        for cx, cy, cr in candidates:
            if all((cx - ux) ** 2 + (cy - uy) ** 2 > (min(cr, ur) * 0.7) ** 2 for ux, uy, ur in unique_candidates):
                unique_candidates.append((cx, cy, cr))

        holes = self._verify_candidates(unique_candidates, depth_mm, depression, object_mask)
        debug = {
            "object_mask": object_mask,
            "depression": depression,
            "candidate_count": len(unique_candidates),
            "px_per_mm": px_per_mm,
        }
        return holes, debug

    def _verify_candidates(self, candidates, depth_mm, depression, object_mask):
        cfg = self.config
        height, width = depth_mm.shape
        holes: list[DetectedHole] = []
        for x, y, radius in candidates:
            if not (0 <= x < width and 0 <= y < height) or object_mask[y, x] == 0:
                continue
            evidence = self._circle_depth_evidence(depth_mm, x, y, radius)
            if evidence is None:
                continue
            recess_mm, invalid_fraction, ring_valid_fraction, sector_recesses, ring_depth_mm = evidence

            diameter_mm = (2.0 * radius * ring_depth_mm) / self.intrinsics.fx
            if diameter_mm < cfg.min_hole_diameter_mm or diameter_mm > cfg.max_hole_diameter_mm * 1.5:
                continue
            if self._connects_to_outside_depression(depression, x, y, radius, cfg.connection_depth_mm):
                continue

            edge_ratio = max(abs(x - width / 2) / (width / 2), abs(y - height / 2) / (height / 2))
            is_edge = edge_ratio >= 0.60  # 화면 가장자리는 스테레오 매칭이 덜 완전해서 기준 완화
            required_ring_valid = 0.55 if is_edge else 0.75
            required_sectors = 4 if is_edge else cfg.required_deep_sectors
            required_depth = cfg.min_hole_depth_mm * (0.75 if is_edge else 1.0)

            if ring_valid_fraction < required_ring_valid:
                continue

            deep_sectors = int(np.count_nonzero(sector_recesses >= required_depth))
            accepted = recess_mm is not None and recess_mm <= cfg.max_hole_depth_mm and deep_sectors >= required_sectors
            # 아주 좁고 깊은 홀은 중심부 depth 측정 자체가 실패(무효)할 수 있다 -
            # 그 경우 링이 충분히 유효하고 중심 무효 비율이 높으면 홀로 인정.
            if not accepted and recess_mm is None:
                accepted = invalid_fraction >= 0.65 and ring_valid_fraction >= (0.80 if is_edge else 0.95)
            if not accepted:
                continue

            floor_depth_mm = ring_depth_mm + (recess_mm if recess_mm is not None else cfg.min_hole_depth_mm)
            point = rs_deproject(self.intrinsics, x, y, floor_depth_mm / 1000.0)
            holes.append(DetectedHole(
                pixel=(x, y),
                depth_m=floor_depth_mm / 1000.0,
                hole_depth_mm=recess_mm if recess_mm is not None else cfg.min_hole_depth_mm,
                diameter_px=2.0 * radius,
                point_camera_m=point,
            ))
        return holes


def rs_deproject(intrinsics, u: float, v: float, depth_m: float) -> np.ndarray:
    """[역할] 이미지 픽셀(u, v) + depth 값을 카메라 좌표계 3D 점 (x, y, z)로 변환."""
    import pyrealsense2 as rs
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m))
    return np.array(point, dtype=float)
