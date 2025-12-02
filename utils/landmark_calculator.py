"""
랜드마크 좌표 기반 계산 유틸리티
"""
import numpy as np
from scipy import interpolate


def calculate_length(landmarks, point1_id, point2_id, calc_type):
    """두 점 사이의 거리 계산"""
    try:
        # 점 찾기
        p1 = next((lm for lm in landmarks if lm['mpidx'] == point1_id), None)
        p2 = next((lm for lm in landmarks if lm['mpidx'] == point2_id), None)

        if not p1 or not p2:
            return None

        if calc_type == "직선거리":
            return np.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2 + (p1['z']-p2['z'])**2)
        elif calc_type == "X좌표거리":
            return abs(p1['x'] - p2['x'])
        elif calc_type == "Y좌표거리":
            return abs(p1['y'] - p2['y'])
        else:
            return None

    except Exception as e:
        return None


def calculate_curvature(landmarks, point_ids):
    """점 그룹의 곡률 계산

    Args:
        landmarks: 랜드마크 리스트
        point_ids: 점 번호 리스트 (5-7개)

    Returns:
        각 점에서의 곡률 값 리스트 또는 None
    """
    try:
        if len(point_ids) < 3:
            return None

        # 랜드마크에서 선택된 점들 추출
        selected_points = []
        for point_id in point_ids:
            landmark = next((lm for lm in landmarks if lm['mpidx'] == point_id), None)
            if landmark:
                selected_points.append([landmark['x'], landmark['y']])
            else:
                return None

        if len(selected_points) != len(point_ids):
            return None

        # numpy 배열로 변환
        points = np.array(selected_points)

        # 얼굴 중심 기준으로 방향 정규화 판단
        direction_factor = determine_direction_factor(points, point_ids)

        # parametric t 값 생성 (0부터 점 개수-1까지)
        t = np.arange(len(points))

        # x, y 좌표에 대해 각각 스플라인 보간
        spline_x = interpolate.UnivariateSpline(t, points[:, 0], s=0)
        spline_y = interpolate.UnivariateSpline(t, points[:, 1], s=0)

        # 각 원본 점에서의 곡률 계산
        curvatures = []
        for i in range(len(points)):
            # 1차, 2차 미분 계산
            dx = spline_x.derivative(1)(i)
            dy = spline_y.derivative(1)(i)
            d2x = spline_x.derivative(2)(i)
            d2y = spline_y.derivative(2)(i)

            # 부호 있는 곡률 공식: (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
            # 양수: 위로 볼록(∩), 음수: 아래로 볼록(∪)
            numerator = dx * d2y - dy * d2x
            denominator = (dx**2 + dy**2)**(3/2)

            if denominator == 0:
                curvature = 0
            else:
                curvature = numerator / denominator

            # 방향 정규화 적용
            curvature *= direction_factor

            curvatures.append(curvature)

        return curvatures

    except Exception as e:
        return None


def determine_direction_factor(points, point_ids):
    """얼굴 중심 기준으로 방향 정규화 인수 결정

    Args:
        points: 점들의 좌표 배열 [[x1, y1], [x2, y2], ...]
        point_ids: MediaPipe 점 번호들

    Returns:
        1 또는 -1 (방향 정규화 인수)
    """

    # 얼굴 중심 X 좌표 (대략 200-250 범위, 이미지 너비 500 기준)
    face_center_x = 250

    # 시작점과 끝점의 X 좌표
    start_x = points[0][0]
    end_x = points[-1][0]

    # 전체 이동 방향 (내측→외측 기준)
    overall_direction = end_x - start_x

    # 좌측/우측 판단
    avg_x = np.mean(points[:, 0])
    is_left_side = avg_x < face_center_x

    # 방향 정규화 로직
    if is_left_side:
        # 좌측: 내측→외측이 X 증가 방향 (양수)
        # 정상적인 내측→외측 이동이면 그대로, 반대면 뒤집기
        direction_factor = 1 if overall_direction > 0 else -1
    else:
        # 우측: 내측→외측이 X 감소 방향 (음수)
        # 정상적인 내측→외측 이동이면 뒤집기, 반대면 그대로
        direction_factor = -1 if overall_direction < 0 else 1

    return direction_factor
