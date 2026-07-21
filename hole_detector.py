"""[역할] Depth 기반 홀(스터드/스파이크 구멍) 탐지 알고리즘의 핵심 모듈.

3D 프린트된 타이어 트레드는 검은색 무광 플라스틱이라 RGB만으로는 홀을 찾기
어렵다(대비가 거의 없음). 그래서 이 모듈은 RGB를 쓰지 않고 depth map만으로
홀을 찾는다: 홀은 "주변 트레드 표면보다 카메라에서 더 멀리 떨어진(더 깊은)
국소 영역"이라는 성질을 이용한다.

방법: 예상 홀 지름보다큰 커널로 depth map에 g rayscale morphological
"opening"을 적용하면, 홀처럼 국소적으로 튀어나온(depth 값이 커진) 부분이
지워지고 "홀이 없었다면 이랬을 것"이라는 기준 표면(baseline)이 복원된다.
(타이어 표면이 평면이 아니라 곡면이어도 국소적으로는 잘 동작한다.)
실측 depth와 baseline의 차이(residual)가 큰 곳이 곧 홀이다.

model.py가 이 모듈의 HoleDetector를 호출해서 실시간 프레임마다 홀 목록을
받아온다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

    #카메라 사양 확인 후 값 수정
@dataclass
class HoleDetectorConfig:
    """[역할] 탐지 파라미터 모음. 실제 설치 환경/부품 스펙에 맞게 튜닝하는 곳.

    스터드 핀(압정) 스펙: 머리 지름 11mm / 깊이 3mm (= 카메라가 찾아야 하는
    입구 원), 끝부분 지름 2mm / 깊이 7mm (= 입구 안쪽 깊은 곳이라 D435 해상도
    밖이고 카메라로 볼 대상이 아님 - 로봇이 11mm 입구 중심만 정확히 맞추면 됨).
    """

    # 물체(타이어 출력물) 분리용 거리 범위 - 카메라를 타이어 정면 20~40cm로
    # 옮긴 구도 기준. 이 정도 거리면 D435 depth 노이즈가 1mm 이하로 떨어져서
    # 3mm짜리 홀 깊이와 노이즈가 잘 구분된다.
    min_object_distance_m: float = 0.18
    max_object_distance_m: float = 0.45

    # [진단용 임시값] min만 낮춰서 크기 필터를 느슨하게 함. max는 baseline
    # 커널 크기 계산에도 쓰이므로 원래 스펙(11mm)에 가깝게 유지 - 20mm로
    # 키우면 커널도 같이 커져서 baseline 추정 자체가 왜곡될 수 있음.
    min_hole_diameter_mm: float = 3.0
    max_hole_diameter_mm: float = 15.0

    # 홀 바닥이 주변 표면(baseline)보다 몇 mm 더 깊어야 "진짜 홀"로 인정할지
    # (하한/상한 둘 다). 실측으로는 진짜 홀이 1~2mm 범위로 관측되고, 그보다
    # 큰 값(3mm+)은 V자 띠 등 다른 구조물일 가능성이 높아 상한으로 제외한다.
    min_hole_depth_mm: float = 0.8
    max_hole_depth_mm: float = 2.5

    # [진단용 임시값] 원래 0.55 -> 진짜 홀이 V자 띠와 맞닿아 있어서 컨투어가
    # 합쳐지고 있는지 확인하려고 잠깐 낮춰놓음. 확인되면 다시 0.55 근처로 복구.
    min_circularity: float = 0.3

    # baseline 표면 추정에 쓰는 morphology 커널 크기(px). 0이면 매 프레임마다
    # max_hole_diameter_mm과 현재 거리 기준 px/mm 스케일로 자동 계산.
    baseline_kernel_px: int = 0
    

@dataclass
class DetectedHole:
    """[역할] 탐지된 홀 하나의 정보를 담는 결과 객체."""

    pixel: tuple  # 이미지 상의 (u, v) 중심 좌표 (px)
    depth_m: float  # 홀 바닥의 depth (중앙값), 미터
    hole_depth_mm: float  # 주변 표면 대비 홀이 얼마나 깊은지 (mm)
    diameter_px: float
    point_camera_m: np.ndarray = field(repr=False)  # 카메라 좌표계 (x, y, z), 미터


class HoleDetector:
    """[역할] depth map 한 장을 받아 홀 목록을 반환하는 탐지기 본체."""

    def __init__(self, intrinsics, config: HoleDetectorConfig | None = None):
        self.intrinsics = intrinsics  # 픽셀->3D 변환에 필요한 카메라 내부 파라미터 (rs.intrinsics)
        self.config = config or HoleDetectorConfig()

    def segment_object(self, depth_m: np
                       .ndarray) -> np.ndarray:
        """[역할] 배경을 떼어내고 타이어 출력물만 마스크로 남긴다.

        거리 밴드(min~max_object_distance_m)로 1차 필터링한 뒤, 노  이즈 제거를
        위해 열림/닫힘 연산을 적용하고, 가장 큰 연결 성분(=출력물 본체)만
        남긴다.
        """
        cfg = self.config
        valid = (depth_m > cfg.min_object_distance_m) & (depth_m < cfg.max_object_distance_m)
        mask = valid.astype(np.uint8) * 255
        # 열림 침식 후 팽창으로 작은 노이즈 제거, 5px보다 작은 구멍 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # depth 측정 실패로 작은 구멍이 생기면 팽창이 그 구멍을 메우고 침식이 바깥 경계로 되돌림 2단계에서 바깥 노이즈를 2단계에서 제거, 팽창이 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # 가장 큰 연결 성분만 남김, 흰 덩어리마다 번호를 붙이고 면적을 구함 -> 가장 큰 번호만 남기고 나머지 노이즈 처리
        # connectivity 어느 흰 픽셀들을 하나의 덩어리로 묶을지 4 = 상하좌우 , 8 = 대각선 포함 3^2-1
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)

    def _px_per_mm(self, depth_m: np.ndarray, object_mask: np.ndarray) -> float:
        """[역할] 현재 촬영 거리에서 1mm가 몇 픽셀에 해당하는지 환산.

        홀 지름(mm) 기준값들을 픽셀 단위 면적/커널 크기로 바꾸는 데 쓰인다.
        """
        region_depth = depth_m[object_mask > 0]
        median_depth_m = float(np.median(region_depth)) if region_depth.size else 0.3
        return self.intrinsics.fx / (median_depth_m * 1000.0)

    def detect(self, depth_m: np.ndarray):
        """[역할] 메인 파이프라인: 물체 분리 -> baseline 표면 추정 -> 잔차로
        홀 마스크 생성 -> 개별 홀 추출. (holes 리스트, 디버그용 중간 결과) 반환.
        """
        cfg = self.config
        object_mask = self.segment_object(depth_m)
        if cv2.countNonZero(object_mask) == 0:
            empty = np.zeros_like(object_mask)
            return [], {"object_mask": object_mask, "hole_mask": empty, "residual_mm": None}

        px_per_mm = self._px_per_mm(depth_m, object_mask)
        # baseline 추정 커널은 홀보다 커야 홀을 "지워서" 복원할 수 있다
        kernel_px = cfg.baseline_kernel_px or (max(3, int(round(cfg.max_hole_diameter_mm * px_per_mm))) | 1)

        depth_mm = depth_m * 1000.0
        # 물체 바깥/무효(0) depth는 물체의 최댓값(가장 먼 지점)으로 채워서
        # 배경이나 결측치가 홀로 오검출되지 않게 한다.
        object_depths = depth_mm[(object_mask > 0) & (depth_mm > 0)]
        fill_value = float(np.max(object_depths)) if object_depths.size else 0.0
        working = np.where((object_mask > 0) & (depth_mm > 0), depth_mm, fill_value).astype(np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
        # opening(침식 후 팽창)은 depth map에서 국소적으로 튀어나온(=먼 거리,
        # 즉 홀) 부분을 지우고 주변 표면으로 채워서 "홀이 없다고 가정한"
        # 기준 표면(baseline)을 만들어낸다.
        baseline_mm = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel)

        # 실측값이 baseline보다 멀다(양수) = 그 자리에 구멍이 파여 있다는 뜻
        residual_mm = working - baseline_mm
        residual_mm[object_mask == 0] = 0

        # depth 값이 0(측정 실패)인 픽셀도 강한 홀 신호로 취급한다: 좁고 깊은
        # 홀은 IR 스테레오 매칭 자체가 실패해서 무효값이 찍히는 경우가 많다.
        dropout = (object_mask > 0) & (depth_mm == 0)
        if np.any(dropout):
            fallback = max(cfg.min_hole_depth_mm * 3, float(np.max(residual_mm)) if residual_mm.size else 0.0)
            residual_mm[dropout] = fallback

        # 잔차가 [min, max] 범위 안일 때만 홀 마스크로 확정 (하한=노이즈 제외,
        # 상한=V자 띠처럼 훨씬 깊은 다른 구조물 제외). 자잘한 노이즈는 열림 연산으로 제거
        hole_mask = (
            (residual_mm >= cfg.min_hole_depth_mm)
            & (residual_mm <= cfg.max_hole_depth_mm)
            & (object_mask > 0)
        ).astype(np.uint8) * 255
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        holes = self._extract_holes(hole_mask, depth_m, residual_mm, px_per_mm)
        debug = {
            "object_mask": object_mask,
            "hole_mask": hole_mask,
            "residual_mm": residual_mm,
            "px_per_mm": px_per_mm,
            "kernel_px": kernel_px,  # 진단용: baseline opening에 실제로 쓰인 커널 크기(px)
            "working_mm": working,  # 진단용: object_mask 적용 후 실측 depth(mm)
            "baseline_mm": baseline_mm,  # 진단용: opening으로 복원한 기준 표면(mm)
        }
        return holes, debug

    def _extract_holes(self, hole_mask, depth_m, residual_mm, px_per_mm):
        """[역할] 이진 홀 마스크에서 개별 홀 블롭을 찾아 크기/원형도로
        필터링하고, 각 홀의 중심 픽셀 -> 3D 좌표까지 계산해 DetectedHole로 반환.
        """
        cfg = self.config
        contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area_px = np.pi * (cfg.min_hole_diameter_mm / 2 * px_per_mm) ** 2
        max_area_px = np.pi * (cfg.max_hole_diameter_mm / 2 * px_per_mm) ** 2 * 2.5  # 여유있게 상한 설정

        holes: list[DetectedHole] = []
        for contour in contours:
            # 1) 면적 필터: 설계된 홀 크기 범위 밖이면 제외
            area = cv2.contourArea(contour)
            if area < min_area_px or area > max_area_px:
                continue
            # 2) 원형도 필터: 홀은 원형에 가까워야 함, 길쭉한 잡음/모서리 제외
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < cfg.min_circularity:
                continue

            single_mask = np.zeros(hole_mask.shape, dtype=np.uint8)
            cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # 홀 하나의 픽셀 중심(무게중심) 계산
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]

            # 해당 홀 영역의 실측 depth 중앙값 -> 3D 좌표 계산에 사용
            region_depth = depth_m[single_mask > 0]
            region_depth = region_depth[region_depth > 0]
            if region_depth.size == 0:
                continue
            hole_depth_m = float(np.median(region_depth))
            hole_depth_mm = float(np.median(residual_mm[single_mask > 0]))
            diameter_px = 2 * np.sqrt(area / np.pi)

            # 픽셀 좌표 + depth -> 카메라 좌표계 3D 점으로 역투영(deproject)
            point = rs_deproject(self.intrinsics, u, v, hole_depth_m)
            holes.append(DetectedHole(
                pixel=(int(round(u)), int(round(v))),
                depth_m=hole_depth_m,
                hole_depth_mm=hole_depth_mm,
                diameter_px=diameter_px,
                point_camera_m=point,
            ))
        return holes


def rs_deproject(intrinsics, u: float, v: float, depth_m: float) -> np.ndarray:
    """[역할] 이미지 픽셀(u, v) + depth 값을 카메라 좌표계 3D 점 (x, y, z)로 변환."""
    import pyrealsense2 as rs
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m))
    return np.array(point, dtype=float)
