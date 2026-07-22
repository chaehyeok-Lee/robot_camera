"""[역할] 거리별 depth 노이즈를 실측해서, 이 카메라/이 물체/이 환경에서 정확도가
가장 좋을 거리를 데이터로 찾는 진단 스크립트.

스펙 시트("2m에서 오차 2%")는 일반적인 수치일 뿐, 우리 환경(조명, 표면 재질,
물체 각도)에서 실제로 몇 cm가 제일 깨끗한지는 직접 재봐야 안다. 이 스크립트는
지정한 지점의 depth가 프레임마다 얼마나 흔들리는지(시간축 노이즈)를 여러 거리에서
측정해서, 거리 대 노이즈 그래프를 그려준다.

사용법:
    python distance_calibration.py

조작:
    (미리보기에서) 표면의 평평한 지점을 클릭 -> 측정 기준점 설정 (자홍색 십자로 표시)
    스페이스바 - 지금 위치에서 30프레임 측정 (카메라/물체를 다음 거리로 옮기고 다시 누르기 반복)
    q - 측정 종료 -> 그래프 저장 + 최적 거리 출력
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pyrealsense2 as rs

matplotlib.rcParams["font.family"] = "Malgun Gothic"  # Windows 기본 한글 폰트 (안 그러면 그래프 글자 깨짐)
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, depth_sensor.get_option_range(rs.option.laser_power).max)
    except RuntimeError:
        pass
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    click_state = {"pixel": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["pixel"] = (x, y)

    cv2.namedWindow("Distance calibration")
    cv2.setMouseCallback("Distance calibration", on_mouse)

    results: list[tuple[float, float, float, int]] = []  # (distance_mm, temporal_noise_mm, spatial_noise_mm, n)

    print(
        "평평한 지점을 클릭해서 기준점을 잡고, 스페이스바로 측정하세요. "
        "측정마다 카메라/물체를 다음 거리로 옮기고 다시 스페이스바. 끝나면 q."
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()
            if click_state["pixel"] is not None:
                cv2.drawMarker(display, click_state["pixel"], (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, f"measurements: {len(results)}  (space=measure, q=finish)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Distance calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if click_state["pixel"] is None:
                    print("먼저 평평한 지점을 클릭해서 기준점을 잡아주세요.")
                    continue
                px, py = click_state["pixel"]
                print("30프레임 측정 중... 움직이지 마세요.")
                samples_mm = []
                for _ in range(30):
                    f = pipeline.wait_for_frames()
                    f = align.process(f)
                    df = f.get_depth_frame()
                    if not df:
                        continue
                    d = df.get_distance(px, py) * 1000.0  # mm, 0이면 무효
                    if d > 0:
                        samples_mm.append(d)
                if len(samples_mm) < 5:
                    print("유효한 depth가 너무 적습니다 (그 지점이 범위 밖이거나 무효값일 수 있음) - 다시 시도하세요.")
                    continue

                distance_mm = float(np.median(samples_mm))
                temporal_noise_mm = float(np.std(samples_mm))  # 같은 픽셀이 프레임마다 얼마나 흔들리는지

                # depth가 1mm 단위로만 나오므로, 30프레임이 우연히 전부 같은 정수값으로
                # 찍히면 std가 정확히 0.0이 될 수 있다 - 이건 "노이즈가 없다"가 아니라
                # "우리 측정 해상도(1mm)로는 이보다 작은 노이즈를 못 잰다"는 뜻이라 믿을 수
                # 없는 값이다. 자동으로 버리고 다시 측정하게 한다.
                if temporal_noise_mm == 0.0:
                    print(
                        "노이즈가 정확히 0.000mm로 나왔습니다 - depth가 1mm 단위로만 찍혀서 "
                        "생기는 우연이라 이 값은 못 믿습니다. 이 측정은 버리고 다시 측정하세요."
                    )
                    continue

                # 공간적 노이즈: 기준점 주변 5x5의 depth 편차 (표면이 평평하다고 가정)
                depth_image_mm = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale * 1000.0
                patch = depth_image_mm[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
                patch_valid = patch[patch > 0]
                spatial_noise_mm = float(np.std(patch_valid)) if patch_valid.size >= 5 else float("nan")

                results.append((distance_mm, temporal_noise_mm, spatial_noise_mm, len(samples_mm)))
                print(
                    f"[측정 {len(results)}] 거리={distance_mm:.1f}mm  "
                    f"시간축 노이즈(std)={temporal_noise_mm:.3f}mm  공간 노이즈(std)={spatial_noise_mm:.3f}mm"
                )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # 방어적으로 한 번 더 필터링 (혹시 모를 0.0 값 제거)
    results = [r for r in results if r[1] > 0.0]
    if len(results) < 2:
        print("측정이 2개 미만이라 그래프를 만들 수 없습니다.")
        return

    results.sort(key=lambda r: r[0])
    out_dir = Path("distance_calibration_results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / f"measurements_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["distance_mm", "temporal_noise_mm", "spatial_noise_mm", "n_valid_frames"])
        writer.writerows(results)
    print(f"CSV 저장: {csv_path}")

    distances = [r[0] for r in results]
    temporal = [r[1] for r in results]
    spatial = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, temporal, "o-", label="시간축 노이즈 (프레임 간 흔들림, mm)", color="tab:red")
    ax.plot(distances, spatial, "s--", label="공간 노이즈 (주변 5x5 편차, mm)", color="tab:blue")
    best_index = int(np.argmin(temporal))
    ax.axvline(distances[best_index], color="gray", linestyle=":", alpha=0.7)
    ax.annotate(
        f"최적: {distances[best_index]:.0f}mm\n(노이즈 {temporal[best_index]:.2f}mm)",
        (distances[best_index], temporal[best_index]),
        textcoords="offset points", xytext=(10, 10),
    )
    ax.set_xlabel("거리 (mm)")
    ax.set_ylabel("depth 노이즈 (mm, 낮을수록 좋음)")
    ax.set_title("거리별 depth 노이즈 실측")
    ax.legend()
    ax.grid(alpha=0.3)
    png_path = out_dir / f"plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"그래프 저장: {png_path}")

    print(
        f"\n[결론] 지금까지 측정한 {len(results)}개 지점 중 "
        f"거리 {distances[best_index]:.0f}mm(≈{distances[best_index] / 10:.1f}cm)에서 "
        f"노이즈가 가장 낮았습니다 (시간축 std {temporal[best_index]:.3f}mm)."
    )
    print("참고: 우리가 찾는 홀 신호(1~2mm)보다 이 노이즈 값이 확실히 작아야 안정적으로 탐지됩니다.")


if __name__ == "__main__":
    main()
