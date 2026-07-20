"""[역할] 카메라 좌표계 <-> 로봇 좌표계 변환 담당 모듈.

hole_detector.py가 찾은 홀의 3D 좌표는 D435 카메라 기준(camera-optical frame)
좌표계로 나온다. 실제로 로봇이 스터드를 박으려면 이 좌표를 로봇 베이스 좌표계로
바꿔야 하는데, 그 변환에 쓰이는 4x4 동차 변환 행렬(hand-eye calibration 결과)을
파일에서 읽어오고, 점 하나를 실제로 변환하는 함수를 제공한다.

hand_eye_calibration.json이 아직 없으면(=핸드아이 캘리브레이션을 아직 안 했으면)
identity 행렬을 반환해서 카메라 좌표를 그대로 통과시키고 경고를 출력한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 실제 캘리브레이션 결과가 저장될 파일 경로 (없으면 identity로 대체됨)
DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "hand_eye_calibration.json"


def load_camera_to_robot(path: Path = DEFAULT_CALIBRATION_PATH) -> np.ndarray:
    """hand_eye_calibration.json에서 4x4 camera->robot 변환 행렬을 읽는다.

    파일이 없으면(캘리브레이션 전) identity를 반환 -> 이후 좌표는 로봇 좌표가
    아니라 카메라 좌표 그대로 나간다는 뜻이므로 반드시 경고를 띄운다.
    """
    if not path.exists():
        print(
            f"[calibration] {path.name} not found - returning identity. "
            "Hole coordinates will be in the CAMERA frame, not the robot frame. "
            "Run hand-eye calibration and save the 4x4 matrix to this file "
            "(see hand_eye_calibration.example.json) to fix."
        )
        return np.eye(4)

    data = json.loads(path.read_text())
    matrix = np.array(data["camera_to_robot"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path} must contain a 4x4 'camera_to_robot' matrix")
    return matrix


def transform_point(point_camera_m: np.ndarray, camera_to_robot: np.ndarray) -> np.ndarray:
    """카메라 좌표계의 점 하나(x, y, z[m])를 로봇 베이스 좌표계로 변환한다."""
    homogeneous = np.append(point_camera_m, 1.0)  # 동차좌표로 확장 (x,y,z,1)
    robot_point = camera_to_robot @ homogeneous
    return robot_point[:3]
