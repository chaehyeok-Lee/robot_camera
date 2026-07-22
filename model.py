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
    r - 실시간 화면의 안정화 트래커 초기화 (홀이 잘못 고정돼 안 없어질 때)

실시간 화면의 빨간 원은 매 프레임 새로 계산한 게 아니라 두 겹으로 안정화된다:
1) 최근 15프레임을 모아 픽셀별 중앙값으로 depth 노이즈를 줄인 뒤 그 결과로 탐지
   (실측 결과 진짜 홀의 depth 신호가 프레임마다 0~2.5mm로 크게 흔들려서, 단일
   프레임 판정으로는 임계값을 못 넘기는 경우가 많았다)
2) 그 위에 HoleTracker로 몇 프레임 연속 잡혀야 화면에 나타나게 함
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration import load_camera_to_robot, transform_point
from hole_detector import HoleDetector, HoleDetectorConfig, HoleTracker


def main() -> None:
    config = HoleDetectorConfig()
    # 캘리브레이션 파일이 없으면 identity(카메라 좌표 그대로) - calibration.py 참고
    camera_to_robot = load_camera_to_robot()
    # 매 프레임 독립 탐지 결과가 깜빡이지 않도록 안정화 - 여러 프레임 연속으로
    # 잡혀야 화면에 표시되고, 잠깐 놓쳐도 바로 안 사라짐.
    tracker = HoleTracker(confirm_frames=5, miss_tolerance=10)

    # 실시간용 롤링 노이즈 감소 버퍼 - 단일 프레임 depth는 노이즈가 커서 진짜 홀
    # 신호(1~2mm)가 프레임마다 0~2.5mm로 흔들리는 게 실측으로 확인됐다. 최근
    # ROLLING_WINDOW개 프레임을 계속 모아 픽셀별 중앙값으로 판정하면 훨씬
    # 안정적이다 (15프레임 = 약 0.5초 지연, 정지된 물체 검사라 문제없음).
    ROLLING_WINDOW = 15
    depth_history: deque = deque(maxlen=ROLLING_WINDOW)  # 원본(raw, 스케일 전) depth
    color_history: deque = deque(maxlen=ROLLING_WINDOW)

    def fuse_history() -> tuple[np.ndarray, np.ndarray] | None:
        if not depth_history:
            return None
        stacked = np.stack([np.where(d > 0, d, np.nan) for d in depth_history], axis=0)
        with np.errstate(invalid="ignore"):
            fused_raw = np.nanmedian(stacked, axis=0)
        fused_depth_m = np.nan_to_num(fused_raw, nan=0.0) * depth_scale
        fused_color = np.median(np.stack(color_history, axis=0), axis=0).astype(np.uint8)
        return fused_depth_m, fused_color

    # --- 1. RealSense 파이프라인 설정: depth + color 스트림 시작 ---
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(rs_config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy 프리셋 (정밀도 우선)
    # IR 텍스처 프로젝터를 최대 출력으로 - 무광 검은 표면은 IR 반사가 약해서
    # depth 노이즈/무효 비율이 높은데, 액티브 스테레오 프로젝터를 강하게 켜면
    # 매칭용 텍스처가 늘어나 신호 품질이 개선된다.
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
        if depth_sensor.supports(rs.option.laser_power):
            max_power = depth_sensor.get_option_range(rs.option.laser_power).max
            depth_sensor.set_option(rs.option.laser_power, max_power)
            print(f"IR 프로젝터 활성화 (레이저 파워 {max_power:.0f})")
    except RuntimeError as error:
        print(f"IR 프로젝터 설정 실패 (일부 펌웨어는 스트리밍 중 변경 불가): {error}")
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
        if debug is not None and debug.get("depression") is not None:
            in_object = bool(debug["object_mask"][py, px] > 0)
            # depression 인코딩은 1단위 ≈ 0.5mm (hole_detector._make_depression_image 참고)
            depression_mm = float(debug["depression"][py, px]) / 2.0
            px_per_mm = debug.get("px_per_mm")
            candidate_count = debug.get("candidate_count")
            residual_note = (
                f"  object_mask={'안' if in_object else '밖'}  depression≈{depression_mm:.2f}mm"
                f"  px_per_mm={px_per_mm:.2f}  candidates={candidate_count}"
            )

        print(
            f"[클릭] pixel=({px},{py})  depth={d:.3f} m ({d * 1000:.0f} mm)  "
            f"{'범위 안' if in_band else '범위 밖'} (현재 설정 {config.min_object_distance_m}~{config.max_object_distance_m}m)"
            f"{residual_note}"
        )

    cv2.namedWindow("RGB + holes | Depth")
    cv2.setMouseCallback("RGB + holes | Depth", on_mouse, click_state)

    def capture_averaged(n_frames: int = 30) -> tuple[np.ndarray, np.ndarray] | None:
        """[역할] n_frames만큼 depth+color를 모아 픽셀별 중앙값을 반환 - 단일 프레임의
        랜덤 노이즈를 줄여서 얕은 홀(1~2mm급) 신호를 더 안정적으로 살리기 위함.
        depth의 무효(0) 픽셀은 NaN으로 바꿔 nanmedian으로 제외한다 - 그냥 median을
        쓰면 dropout(0)이 섞여 중앙값이 왜곡될 수 있다.
        캡처 도중 물체/카메라가 움직이면 오히려 결과가 흐려지므로 정지 상태 필요.
        """
        depth_stack, color_stack = [], []
        for _ in range(n_frames):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_frame = temporal.process(depth_frame)
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_stack.append(np.where(depth_raw > 0, depth_raw, np.nan))
            color_stack.append(np.asanyarray(color_frame.get_data()))
        if not depth_stack:
            return None
        with np.errstate(invalid="ignore"):
            fused_depth = np.nanmedian(np.stack(depth_stack, axis=0), axis=0)
        fused_depth = np.nan_to_num(fused_depth, nan=0.0) * depth_scale
        fused_color = np.median(np.stack(color_stack, axis=0), axis=0).astype(np.uint8)
        return fused_depth, fused_color

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
        # depression: "국소 표면보다 얼마나 더 먼가"를 밝기로 표현한 디버그 이미지.
        # 여기서 Hough Circle 후보를 찾으므로, 밝게 뭉친 부분이 곧 후보 위치다.
        depression_vis = cv2.applyColorMap(debug["depression"], cv2.COLORMAP_TURBO)

        cv2.imshow("RGB + holes | Depth", np.hstack([display, depth_vis]))
        cv2.imshow("Depth depression (Hough 입력)", depression_vis)

    print(
        "D435 hole inspection running. 'q' to quit, 's' to save, 'c' to capture "
        "a 30-frame averaged (denoised) detection, 'r' to reset the live tracker, "
        "click preview to read depth."
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

            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # 스케일 전 원본 - 버퍼용
            color_image = np.asanyarray(color_frame.get_data())
            depth_history.append(depth_raw)
            color_history.append(color_image)

            # 최근 프레임들을 픽셀별 중앙값으로 합쳐 노이즈를 줄인 뒤 그걸로 탐지
            depth_m, fused_color = fuse_history()
            last_color = fused_color
            click_state["depth_m"] = depth_m  # on_mouse가 최신(융합된) 프레임을 읽도록 갱신

            # 핵심 탐지 호출 (hole_detector.py) - RGB(Hough용) + depth 둘 다 전달
            raw_holes, debug = detector.detect(fused_color, depth_m)
            # 매 프레임 결과를 바로 쓰지 않고 트래커에 통과시켜 깜빡임을 없앤다 -
            # 여러 프레임 연속으로 잡힌 것만 화면에 남고, 잠깐 놓쳐도 유지된다.
            holes = tracker.update(raw_holes)
            last_holes = holes  # 's' 키로 저장할 때 쓰기 위해 보관
            click_state["debug"] = debug  # on_mouse가 object_mask/depression도 읽도록 갱신

            render(depth_m, fused_color, holes, debug, frozen=False)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_holes(last_holes, camera_to_robot)
            if key == ord("r"):
                tracker.clear()
                depth_history.clear()
                color_history.clear()
                print("트래커 + 롤링 버퍼 초기화 - 안정화된 홀 목록을 비웠습니다.")
            if key == ord("c"):
                # --- 다중 프레임 평균 캡처: 물체를 고정한 채 노이즈를 줄여 재탐지 ---
                while True:
                    print("30프레임 캡처 중... 물체/카메라를 움직이지 마세요.")
                    captured = capture_averaged(n_frames=30)
                    if captured is None:
                        print("캡처 실패 - 다시 시도하세요.")
                        break
                    avg_depth_m, avg_color = captured
                    avg_holes, avg_debug = detector.detect(avg_color, avg_depth_m)
                    last_holes = avg_holes
                    last_color = avg_color
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
