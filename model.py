"""[역할] 메인 실행 스크립트 - D435 실시간 캡처부터 로봇 좌표 저장까지 전체 파이프라인.

흐름: RealSense D435로 depth+color 스트림을 받아 depth를 color 시점에
정렬(align)하고 -> hole_detector.HoleDetector로 매 프레임 홀을 탐지 ->
결과를 화면에 오버레이로 표시 -> 's' 키를 누르면 홀 좌표(카메라/로봇 기준)를
JSON 파일로 저장.

3D 프린트된(검은 플라스틱) 타이어 트레드의 스터드/스파이크 홀을 못 박기 전에
검사하는 용도. RGB 대비가 낮아 depth 채널 위주로 판단한다.

사용법:
    python model.py

조작 (미리보기 창에 포커스가 있는 상태에서):
    q - 종료
    s - 현재 프레임에서 탐지된 홀 좌표를 holes_<timestamp>.json으로 저장
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration import load_camera_to_robot, transform_point
from hole_detector import HoleDetector, HoleDetectorConfig


def main() -> None:
    config = HoleDetectorConfig()
    # 캘리브레이션 파일이 없으면 identity(카메라 좌표 그대로) - calibration.py 참고
    camera_to_robot = load_camera_to_robot()

    # --- 1. RealSense 파이프라인 설정: depth + color 스트림 시작 ---
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(rs_config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy 프리셋 (정밀도 우선)
    depth_scale = depth_sensor.get_depth_scale()  # raw depth 값(uint16) -> 미터 환산 계수

    # depth를 color 카메라 시점으로 정렬 (두 센서 위치가 달라서 필요)
    align = rs.align(rs.stream.color)
    # depth 노이즈 억제 필터 (hole-filling 필터는 진짜 홀까지 메워버리므로 사용하지 않음)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 8)
    temporal = rs.temporal_filter()

    # 픽셀 -> 3D 좌표 역투영에 필요한 카메라 내부 파라미터
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    detector = HoleDetector(color_intrinsics, config)

    # 미리보기 창을 클릭하면 그 지점의 실제 depth(m/mm)를 콘솔에 출력 -
    # "저 물체가 카메라에서 몇 m 거리인지" 눈대중 대신 바로 재기 위한 디버그 도구.
    # 창은 [RGB | depth 컬러맵]을 가로로 붙여놨으니, 오른쪽 절반을 클릭해도
    # 같은 depth_m 배열에서 좌표만 옮겨서 읽는다.
    click_state = {"depth_m": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or param["depth_m"] is None:
            return
        depth_snapshot = param["depth_m"]
        height, width = depth_snapshot.shape
        px, py = (x - width if x >= width else x), y
        if not (0 <= px < width and 0 <= py < height):
            return
        d = float(depth_snapshot[py, px])
        in_band = config.min_object_distance_m < d < config.max_object_distance_m
        print(
            f"[클릭] pixel=({px},{py})  depth={d:.3f} m ({d * 1000:.0f} mm)  "
            f"{'범위 안' if in_band else '범위 밖'} (현재 설정 {config.min_object_distance_m}~{config.max_object_distance_m}m)"
        )

    cv2.namedWindow("RGB + holes | Depth")
    cv2.setMouseCallback("RGB + holes | Depth", on_mouse, click_state)

    print("D435 hole inspection running. 'q' to quit, 's' to save detected coordinates, click preview to read depth.")
    last_holes: list = []
    try:
        # --- 2. 메인 루프: 프레임마다 캡처 -> 정렬 -> 탐지 -> 시각화 ---
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            click_state["depth_m"] = depth_m  # on_mouse가 최신 프레임을 읽도록 갱신

            # 핵심 탐지 호출 (hole_detector.py)
            holes, debug = detector.detect(depth_m)
            last_holes = holes  # 's' 키로 저장할 때 쓰기 위해 보관

            # --- 3. 시각화: RGB 위에 탐지된 홀을 원+라벨로 표시 ---
            display = color_image.copy()
            for i, hole in enumerate(holes):
                u, v = hole.pixel
                cv2.circle(display, (u, v), max(3, int(hole.diameter_px / 2)), (0, 0, 255), 2)
                cv2.putText(
                    display, f"#{i} d={hole.hole_depth_mm:.1f}mm", (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )
            cv2.putText(display, f"holes: {len(holes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # depth를 색으로 보여주는 창 (노이즈 수준/거리 튜닝할 때 참고용)
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_m, alpha=255.0 / config.max_object_distance_m), cv2.COLORMAP_JET
            )
            # 탐지 파이프라인 중간 산출물인 이진 홀 마스크 (디버깅용)
            hole_mask_vis = cv2.cvtColor(debug["hole_mask"], cv2.COLOR_GRAY2BGR)

            cv2.imshow("RGB + holes | Depth", np.hstack([display, depth_vis]))
            cv2.imshow("Hole mask", hole_mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_holes(last_holes, camera_to_robot)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def save_holes(holes, camera_to_robot: np.ndarray) -> None:
    """[역할] 현재 탐지된 홀들을 카메라 좌표 + 로봇 좌표(변환 적용)로 JSON 저장."""
    payload = []
    for i, hole in enumerate(holes):
        robot_point = transform_point(hole.point_camera_m, camera_to_robot)
        payload.append({
            "id": i,
            "pixel": list(hole.pixel),
            "camera_point_m": hole.point_camera_m.tolist(),
            "robot_point_m": robot_point.tolist(),
            "hole_depth_mm": hole.hole_depth_mm,
        })
    out_path = Path(f"holes_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(payload)} holes -> {out_path}")


if __name__ == "__main__":
    main()
