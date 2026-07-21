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
    s - 현재 탐지된 홀 좌표를 holes_<timestamp>.json으로 저장
    c - 30프레임을 모아 중앙값으로 노이즈를 줄인 뒤 그 결과로 탐지 (검사용 안정된 판정).
        캡처 중엔 물체/카메라를 움직이지 말 것. 결과 화면은 고정되며, 아무 키나
        누르면 실시간 미리보기로 돌아감 (c/s/q는 각자 원래 동작 유지)
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
    click_state = {"depth_m": None, "debug": None, "last_click_raw": None, "last_click": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or param["depth_m"] is None:
            return
        depth_snapshot = param["depth_m"]
        height, width = depth_snapshot.shape
        px, py = (x - width if x >= width else x), y
        param["last_click_raw"] = (x, y)  # cv2가 실제로 보고한 원본 좌표 (창 크기/DPI 확인용)
        if not (0 <= px < width and 0 <= py < height):
            param["last_click"] = None
            print(f"[클릭] 원본좌표=({x},{y})  -> 배열 범위 밖이라 무시됨 (이미지 크기 {width}x{height})")
            return
        param["last_click"] = (px, py)
        d = float(depth_snapshot[py, px])
        in_band = config.min_object_distance_m < d < config.max_object_distance_m

        debug = param["debug"]
        residual_note = ""
        if debug is not None and debug.get("residual_mm") is not None:
            in_object = bool(debug["object_mask"][py, px] > 0)
            residual_val = float(debug["residual_mm"][py, px])
            kernel_px = debug.get("kernel_px")
            px_per_mm = debug.get("px_per_mm")
            working_val = float(debug["working_mm"][py, px])
            baseline_val = float(debug["baseline_mm"][py, px])
            residual_note = (
                f"  residual={residual_val:.2f}mm  object_mask={'안' if in_object else '밖'}"
                f"  kernel={kernel_px}px  px_per_mm={px_per_mm:.2f}"
                f"  working={working_val:.1f}mm  baseline={baseline_val:.1f}mm"
            )

        print(
            f"[클릭] pixel=({px},{py})  depth={d:.3f} m ({d * 1000:.0f} mm)  "
            f"{'범위 안' if in_band else '범위 밖'} (현재 설정 {config.min_object_distance_m}~{config.max_object_distance_m}m)"
            f"{residual_note}"
        )

    cv2.namedWindow("RGB + holes | Depth")
    cv2.setMouseCallback("RGB + holes | Depth", on_mouse, click_state)

    def capture_averaged_depth(n_frames: int = 30) -> np.ndarray | None:
        """[역할] n_frames만큼 depth를 모아 픽셀별 중앙값을 반환 - 단일 프레임의
        랜덤 노이즈를 줄여서 얕은 홀(3mm급) 신호를 더 안정적으로 살리기 위함.
        캡처 도중 물체/카메라가 움직이면 오히려 결과가 흐려지므로 정지 상태 필요.
        """
        stack = []
        for _ in range(n_frames):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_frame = temporal.process(depth_frame)
            stack.append(np.asanyarray(depth_frame.get_data()).astype(np.float32))
        if not stack:
            return None
        return np.median(np.stack(stack, axis=0), axis=0) * depth_scale

    def render(depth_m: np.ndarray, color_image: np.ndarray, holes: list, debug: dict, frozen: bool) -> None:
        """[역할] 한 프레임(실시간 또는 평균 캡처 결과)을 오버레이 그려서 창에 표시."""
        display = color_image.copy()
        for i, hole in enumerate(holes):
            u, v = hole.pixel
            cv2.circle(display, (u, v), max(3, int(hole.diameter_px / 2)), (0, 0, 255), 2)
            cv2.putText(
                display, f"#{i} d={hole.hole_depth_mm:.1f}mm", (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
        cv2.putText(display, f"holes: {len(holes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if frozen:
            cv2.putText(display, "AVERAGED (30f) - press any key for live, c to recapture",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        if click_state["last_click"] is not None:
            cx, cy = click_state["last_click"]
            cv2.drawMarker(display, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_m, alpha=255.0 / config.max_object_distance_m), cv2.COLORMAP_JET
        )
        hole_mask_vis = cv2.cvtColor(debug["hole_mask"], cv2.COLOR_GRAY2BGR)

        cv2.imshow("RGB + holes | Depth", np.hstack([display, depth_vis]))
        cv2.imshow("Hole mask", hole_mask_vis)

    print(
        "D435 hole inspection running. 'q' to quit, 's' to save, 'c' to capture "
        "a 30-frame averaged (denoised) detection, click preview to read depth."
    )
    last_holes: list = []
    last_color: np.ndarray | None = None
    try:
        # --- 2. 메인 루프: 프레임마다 캡처 -> 정렬 -> 탐지 -> 시각화 ---
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # spatial filter는 3mm짜리 얕은 홀 신호까지 뭉개버리는 게 확인돼서 계속
            # 끔. temporal filter는 같은 픽셀을 시간축으로만 평균내서 옆 픽셀(=홀
            # 모양)을 섞지 않으므로 다시 사용.
            # depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            last_color = color_image
            click_state["depth_m"] = depth_m  # on_mouse가 최신 프레임을 읽도록 갱신

            # 핵심 탐지 호출 (hole_detector.py)
            holes, debug = detector.detect(depth_m)
            last_holes = holes  # 's' 키로 저장할 때 쓰기 위해 보관
            click_state["debug"] = debug  # on_mouse가 residual/object_mask도 읽도록 갱신

            render(depth_m, color_image, holes, debug, frozen=False)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_holes(last_holes, camera_to_robot)
            if key == ord("c"):
                # --- 다중 프레임 평균 캡처: 물체를 고정한 채 노이즈를 줄여 재탐지 ---
                while True:
                    print("30프레임 캡처 중... 물체/카메라를 움직이지 마세요.")
                    avg_depth_m = capture_averaged_depth(n_frames=30)
                    if avg_depth_m is None:
                        print("캡처 실패 - 다시 시도하세요.")
                        break
                    avg_holes, avg_debug = detector.detect(avg_depth_m)
                    last_holes = avg_holes
                    click_state["depth_m"] = avg_depth_m
                    click_state["debug"] = avg_debug
                    print(f"평균 캡처 완료 - 홀 {len(avg_holes)}개 탐지됨 (s로 저장, c로 재촬영, 다른 키는 실시간 복귀)")

                    render(avg_depth_m, last_color, avg_holes, avg_debug, frozen=True)
                    review_key = cv2.waitKey(0) & 0xFF
                    if review_key == ord("q"):
                        return
                    if review_key == ord("s"):
                        save_holes(last_holes, camera_to_robot)
                    if review_key == ord("c"):
                        continue  # 재촬영
                    break  # 그 외 키 -> 실시간 루프로 복귀
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
