"""
Face Coordinate Analyzer
실시간 좌표 계산 기반 얼굴 분석 플랫폼
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

# Database (back_analysis)
import sys
import os
sys.path.insert(0, "/home/wavus/새 폴더/back_analysis/src")

from database.connection import DatabaseManager
from database.crud import crud_service

# Initialize db_manager
db_manager = DatabaseManager()

# Utils modules
from utils.landmark_calculator import calculate_landmarks_metric, calculate_length
from utils.data_analyzer import execute_length_based_analysis
from utils.tag_processor import (
    get_tag_groups,
    analyze_tag_relationships,
    execute_single_tag_analysis,
    execute_level_comparison_analysis,
    execute_level_comparison_analysis_ratio,
    execute_level_curvature_analysis
)
from utils.visualization import create_sankey_diagram
from utils.statistical_analyzer import (
    get_available_atomic_measurements,
    get_available_derived_measurements,
    get_available_tags,
    prepare_statistical_dataset,
    recommend_statistical_methods,
    analyze_1to1_pearson,
    analyze_1to1_spearman,
    analyze_1to1_ttest,
    visualize_1to1,
    analyze_2to1_regression,
    analyze_2to1_anova,
    visualize_2to1,
    analyze_3to1_random_forest,
    visualize_3to1,
    analyze_4plus_pca,
    analyze_4plus_tsne,
    visualize_4plus
)

# Page config
st.set_page_config(
    page_title="Face Coordinate Analyzer",
    page_icon="🎭",
    layout="wide"
)


def main():
    st.title("🎭 Face Coordinate Analyzer")

    # 사이드바에 데이터베이스 관리 기능 추가
    render_database_management_sidebar()

    # 랜드마크 데이터 로드
    landmarks_data = load_landmarks_data()

    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧮 좌표 분석",
        "🔗 태그 연관성 분석",
        "🌊 태그 관계도",
        "📊 태그-수치 분석",
        "🔬 통계적 연관성 분석"
    ])

    with tab1:
        render_landmarks_analysis_tab(landmarks_data)

    with tab2:
        render_tag_analysis_tab(landmarks_data)

    with tab3:
        render_sankey_diagram_tab(landmarks_data)

    with tab4:
        render_tag_analysis_tab_new(landmarks_data)

    with tab5:
        render_statistical_correlation_tab()


def load_landmarks_data():
    """랜드마크 데이터 로드 (DB에서만)"""
    # DB에서 데이터 가져오기
    db_data = crud_service.get_dataframe()

    if db_data.empty:
        st.sidebar.warning("💡 DB에 저장된 데이터가 없습니다.")
        return pd.DataFrame()

    # landmarks 컬럼이 있는 데이터만 필터링
    landmarks_data = db_data[db_data['landmarks'].notna()].copy()

    if landmarks_data.empty:
        st.sidebar.warning("💡 landmarks가 포함된 데이터가 없습니다.")
        return pd.DataFrame()

    return landmarks_data


def render_landmarks_analysis_tab(landmarks_data):
    """좌표 분석 탭 렌더링"""
    st.header("🧮 좌표 분석 (실시간 계산)")
    st.markdown("두 거리를 기반으로 한 비교 분석")

    if landmarks_data.empty:
        st.warning("💡 landmarks가 포함된 JSON 파일이 필요합니다.")
        return

    st.sidebar.success(f"📍 {len(landmarks_data)}개 데이터 로드됨")

    # 1. 계산 목적 선택 (단순화)
    st.sidebar.write("### 1. 계산 목적")
    purpose = st.sidebar.selectbox(
        "분석 목적을 선택하세요:",
        ["📏 거리 측정", "⚖️ 비율 계산", "🌊 곡률 분석"],
        index=1
    )

    # 2. 점 그룹 설정
    if purpose == "🌊 곡률 분석":
        st.sidebar.write("### 2. 점 그룹 설정 (5-7개 점)")
        point_group_input = st.sidebar.text_input(
            "점 번호들 (쉼표로 구분)",
            value="33,161,160,159,158",
            help="예: 33,161,160,159,158 (5개 점)"
        )
        # 점 번호들을 파싱
        try:
            l1_points = [int(x.strip()) for x in point_group_input.split(',') if x.strip()]
            if len(l1_points) < 3:
                st.sidebar.error("최소 3개 이상의 점이 필요합니다.")
            elif len(l1_points) > 10:
                st.sidebar.error("최대 10개까지만 입력 가능합니다.")
            else:
                st.sidebar.success(f"{len(l1_points)}개 점 선택됨")
        except:
            st.sidebar.error("올바른 숫자 형식으로 입력하세요.")
            l1_points = [33, 161, 160, 159, 158]

        # 곡률 분석에서는 l1_p1, l1_p2, l1_calc 값을 더미로 설정
        l1_p1, l1_p2 = 0, 1
        l1_calc = "곡률"
    else:
        st.sidebar.write("### 2. 길이1 설정(x축)")
        col1, col2, col3 = st.sidebar.columns([1, 1, 1.2])

        with col1:
            l1_p1 = st.number_input("점1", min_value=0, max_value=500, value=33, key="l1_p1")
        with col2:
            l1_p2 = st.number_input("점2", min_value=0, max_value=500, value=133, key="l1_p2")
        with col3:
            l1_calc = st.selectbox("계산방식", ["직선거리", "X좌표거리", "Y좌표거리"], key="l1_calc")
        l1_points = [l1_p1, l1_p2]

    # 3. 길이2 설정 (비율 계산일 때만)
    if purpose == "⚖️ 비율 계산":
        st.sidebar.write("### 3. 길이2 설정(y축)")
        col1, col2, col3 = st.sidebar.columns([1, 1, 1.2])

        with col1:
            l2_p1 = st.number_input("점1", min_value=0, max_value=500, value=1, key="l2_p1")
        with col2:
            l2_p2 = st.number_input("점2", min_value=0, max_value=500, value=18, key="l2_p2")
        with col3:
            l2_calc = st.selectbox("계산방식", ["직선거리", "X좌표거리", "Y좌표거리"], key="l2_calc")

        # 4. 추가 옵션
        st.sidebar.write("### 4. 추가 옵션")
        normalize_ratio = st.sidebar.checkbox("정규화 (x축=1 고정)", value=True)
        swap_axes = st.sidebar.checkbox("축 바꾸기 (x↔y)")
    else:
        # 거리 측정 또는 곡률 분석일 때는 길이2 설정 불필요
        l2_p1, l2_p2, l2_calc = None, None, None
        normalize_ratio = False
        swap_axes = False

    # 5. 태그 하이라이트 기능
    st.sidebar.write("### 5. 태그 하이라이트")
    enable_tag_highlight = st.sidebar.checkbox("태그별 색상 구분 활성화")

    selected_tags = []
    if enable_tag_highlight:
        # 현재 데이터에서 사용 가능한 태그들 추출
        all_tags = set()
        for _, row in landmarks_data.iterrows():
            if 'tags' in row and row['tags']:
                tags = row['tags'] if isinstance(row['tags'], list) else []
                all_tags.update(tags)

        if all_tags:
            selected_tags = st.sidebar.multiselect(
                "하이라이트할 태그 선택:",
                sorted(list(all_tags)),
                help="선택한 태그를 가진 데이터만 색상으로 표시됩니다."
            )

    # 6. 실행 버튼
    if st.sidebar.button("🔄 분석 실행", type="primary"):
        if purpose == "🌊 곡률 분석":
            # 곡률 분석에서는 l1_points를 추가 파라미터로 전달
            execute_length_based_analysis(
                landmarks_data, l1_p1, l1_p2, l1_calc, l2_p1, l2_p2, l2_calc, purpose,
                normalize_ratio, swap_axes, enable_tag_highlight, selected_tags, l1_points
            )
        else:
            execute_length_based_analysis(
                landmarks_data, l1_p1, l1_p2, l1_calc, l2_p1, l2_p2, l2_calc, purpose,
                normalize_ratio, swap_axes, enable_tag_highlight, selected_tags
            )


def render_tag_analysis_tab(landmarks_data):
    """태그 연관성 분석 탭 렌더링"""
    st.header("🔗 태그 연관성 분석")

    if landmarks_data.empty:
        st.warning("💡 태그가 포함된 데이터가 필요합니다.")
        return

    # 태그 데이터만 필터링
    tag_data = landmarks_data[landmarks_data['tags'].notna()].copy()

    if tag_data.empty:
        st.warning("💡 태그가 포함된 데이터가 없습니다.")
        return

    # 정의된 태그 그룹과 실제 데이터의 태그 비교
    tag_groups = get_tag_groups()
    data_tags = set()
    defined_tags = set()
    for group_tags in tag_groups.values():
        defined_tags.update(group_tags)

    for _, row in tag_data.iterrows():
        if isinstance(row['tags'], list):
            data_tags.update(row['tags'])

    all_unique_tags = sorted(list(data_tags.union(defined_tags)))

    st.write(f"### 📊 태그 현황")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("정의된 태그", len(defined_tags))
    with col2:
        st.metric("데이터 태그", len(data_tags))
    with col3:
        st.metric("전체 고유 태그", len(all_unique_tags))

    # 태그 조합 분석
    st.write("### 🔄 태그 조합 분석")

    # 조합 길이 선택
    combination_length = st.selectbox(
        "분석할 조합 길이:",
        [2, 3, 4, 5],
        index=0
    )

    if st.button("조합 분석 실행"):
        tag_combinations = []

        for _, row in tag_data.iterrows():
            if isinstance(row['tags'], list) and len(row['tags']) >= combination_length:
                # 해당 길이의 모든 조합 생성
                for combo in combinations(row['tags'], combination_length):
                    tag_combinations.append(combo)

        if tag_combinations:
            # 조합 빈도 계산
            combination_counts = Counter(tag_combinations)

            # 상위 조합 표시
            st.write(f"#### 🏆 상위 {combination_length}개 태그 조합")

            top_combinations = combination_counts.most_common(20)
            combo_data = []

            for combo, count in top_combinations:
                combo_data.append({
                    '조합': ' + '.join(combo),
                    '빈도': count,
                    '비율': f"{count/len(tag_data)*100:.1f}%"
                })

            combo_df = pd.DataFrame(combo_data)
            st.dataframe(combo_df, use_container_width=True)

            # 히트맵 생성 (2개 조합인 경우)
            if combination_length == 2 and len(top_combinations) > 5:
                st.write("#### 🌡️ 태그 연관성 히트맵")

                # 상위 태그들 추출
                top_tags = set()
                for combo, count in top_combinations[:15]:  # 상위 15개 조합에서 태그 추출
                    top_tags.update(combo)

                top_tags = sorted(list(top_tags))

                # 히트맵 매트릭스 생성
                matrix = []
                for tag1 in top_tags:
                    row = []
                    for tag2 in top_tags:
                        if tag1 == tag2:
                            count = combination_counts.get((tag1,), 0)  # 자기 자신은 단일 태그 빈도
                        else:
                            # 두 태그의 조합 빈도 (순서 무관)
                            count = combination_counts.get((tag1, tag2), 0) + combination_counts.get((tag2, tag1), 0)
                        row.append(count)
                    matrix.append(row)

                if matrix and len(top_tags) > 1:
                    fig_heatmap = px.imshow(
                        matrix,
                        x=top_tags,
                        y=top_tags,
                        title="태그 간 연관성 강도",
                        labels=dict(color="조합 빈도")
                    )
                    fig_heatmap.update_layout(height=600)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

        else:
            st.warning(f"길이 {combination_length}의 태그 조합이 없습니다.")


def render_sankey_diagram_tab(landmarks_data):
    """Sankey 다이어그램 탭 렌더링"""
    st.header("🌊 태그 관계도 (Sankey Diagram)")

    if landmarks_data.empty:
        st.warning("💡 태그가 포함된 데이터가 필요합니다.")
        return

    # 태그 관계 분석
    relationships = analyze_tag_relationships(landmarks_data)

    if not any(relationships.values()):
        st.warning("💡 태그 관계를 분석할 데이터가 충분하지 않습니다.")
        return

    # 필터 옵션 - 메인 페이지에 배치
    st.write("### 🎛️ 다이어그램 설정")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 관계 타입 선택
        relationship_type = st.selectbox(
            "표시할 관계:",
            ["전체 흐름 (추상→1차→2차)", "추상→1차만", "1차→2차만"]
        )

    with col2:
        # 최소 빈도 설정
        min_frequency = st.slider(
            "최소 관계 빈도:",
            min_value=1,
            max_value=10,
            value=2,
            help="이 빈도 이상의 관계만 표시합니다."
        )

    with col3:
        # 태그 필터 (관계 타입에 따라) - 다중 선택 지원
        if relationship_type in ["전체 흐름 (추상→1차→2차)", "추상→1차만"]:
            selected_abstract_tags = st.multiselect(
                "추상 태그 필터:",
                relationships['abstract_tags'],
                default=[],
                help="빈 선택 시 전체 태그 표시"
            )
            # 빈 선택시 "전체"로 처리
            selected_abstract_tag = selected_abstract_tags if selected_abstract_tags else "전체"
        elif relationship_type == "1차→2차만":
            selected_primary_tags = st.multiselect(
                "1차 태그 필터:",
                relationships['primary_tags'],
                default=[],
                help="빈 선택 시 전체 태그 표시"
            )
            # 빈 선택시 "전체"로 처리
            selected_primary_tag = selected_primary_tags if selected_primary_tags else "전체"
            selected_abstract_tag = "전체"
        else:
            selected_abstract_tag = "전체"
            selected_primary_tag = "전체"

    # 1차→2차만인 경우 selected_primary_tag가 정의되지 않을 수 있으므로 기본값 설정
    if 'selected_primary_tag' not in locals():
        selected_primary_tag = "전체"

    # Sankey 다이어그램 생성
    create_sankey_diagram(
        relationships,
        selected_abstract_tag,
        min_frequency,
        relationship_type,
        selected_primary_tag
    )


def render_tag_analysis_tab_new(landmarks_data):
    """태그-수치 분석 탭 렌더링"""
    st.header("📊 태그-수치 분석")

    if landmarks_data.empty:
        st.warning("💡 landmarks가 포함된 데이터가 필요합니다.")
        return

    # 분석 타입 선택
    analysis_type = st.selectbox(
        "분석 타입 선택:",
        ["🏷️ 단일 태그 분석", "📊 레벨별 비교 분석"]
    )

    if analysis_type == "🏷️ 단일 태그 분석":
        render_single_tag_analysis(landmarks_data, 33, 133, "직선거리")
    else:
        render_level_comparison_analysis(landmarks_data, 33, 133, "직선거리")


def render_single_tag_analysis(landmarks_data, point1, point2, calc_type):
    """단일 태그 분석 렌더링"""
    st.write("### 🏷️ 단일 태그 분석")

    # 사용 가능한 태그 추출
    all_tags = set()
    for _, row in landmarks_data.iterrows():
        if 'tags' in row and row['tags']:
            tags = row['tags'] if isinstance(row['tags'], list) else []
            all_tags.update(tags)

    if not all_tags:
        st.warning("분석할 태그가 없습니다.")
        return

    # 태그 선택
    selected_tag = st.selectbox(
        "분석할 태그 선택:",
        sorted(list(all_tags))
    )

    # 측정 설정
    col1, col2, col3 = st.columns(3)
    with col1:
        point1 = st.number_input("측정점 1", min_value=0, max_value=500, value=point1, step=1, format="%d")
    with col2:
        point2 = st.number_input("측정점 2", min_value=0, max_value=500, value=point2, step=1, format="%d")
    with col3:
        calc_type = st.selectbox("계산 방식", ["직선거리", "X좌표거리", "Y좌표거리"], index=0)

    if st.button("단일 태그 분석 실행"):
        execute_single_tag_analysis(landmarks_data, selected_tag, point1, point2, calc_type)


def render_level_comparison_analysis(landmarks_data, point1, point2, calc_type):
    """레벨별 비교 분석 렌더링"""
    st.write("### 📊 레벨별 비교 분석")

    # 2차 태그에서 특성 추출 (부위-측정값 형태로)
    tag_groups = get_tag_groups()
    features = set()

    for group_name, tags in tag_groups.items():
        if group_name.startswith("2차"):
            for tag in tags:
                if '-' in tag:
                    parts = tag.split('-')
                    if len(parts) >= 3:  # 부위-측정값-레벨 형태
                        feature = f"{parts[0]}-{parts[1]}"  # 예: eye-크기-큰 -> eye-크기
                        features.add(feature)

    if not features:
        st.warning("비교할 2차 태그 특성이 없습니다.")
        return

    # 특성 선택과 측정 방식을 같은 줄에
    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox(
            "비교할 특성:",
            sorted(list(features))
        )
    with col2:
        measurement_type = st.selectbox(
            "측정방식:",
            ["단순 길이", "비율 계산", "곡률 패턴"],
            index=0,
            key="level_measurement_type"
        )

    if measurement_type == "단순 길이":
        col1, col2, col3 = st.columns(3)
        with col1:
            point1 = st.number_input("측정점 1", min_value=0, max_value=500, value=point1, key="level_p1", step=1, format="%d")
        with col2:
            point2 = st.number_input("측정점 2", min_value=0, max_value=500, value=point2, key="level_p2", step=1, format="%d")
        with col3:
            calc_type = st.selectbox("계산 방식", ["직선거리", "X좌표거리", "Y좌표거리"], index=0, key="level_calc")

        if st.button("레벨별 비교 분석 실행", key="level_simple_exec"):
            execute_level_comparison_analysis(landmarks_data, selected_feature, point1, point2, calc_type)

    elif measurement_type == "비율 계산":
        # 분모와 분자를 한 줄에 배치
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 0.5, 1, 1, 1])

        # 분모 설정
        with col1:
            point3 = st.number_input("분모-점1", min_value=0, max_value=500, value=33, key="level_p3_den", step=1, format="%d")
        with col2:
            point4 = st.number_input("분모-점2", min_value=0, max_value=500, value=263, key="level_p4_den", step=1, format="%d")
        with col3:
            calc_type2 = st.selectbox("분모-방식", ["직선거리", "X좌표거리", "Y좌표거리"], index=0, key="level_calc_den")

        with col4:
            st.write("**÷**")

        # 분자 설정
        with col5:
            point1 = st.number_input("분자-점1", min_value=0, max_value=500, value=point1, key="level_p1_num", step=1, format="%d")
        with col6:
            point2 = st.number_input("분자-점2", min_value=0, max_value=500, value=point2, key="level_p2_num", step=1, format="%d")
        with col7:
            calc_type1 = st.selectbox("분자-방식", ["직선거리", "X좌표거리", "Y좌표거리"], index=0, key="level_calc_num")

        if st.button("레벨별 비교 분석 실행 (비율)", key="level_ratio_exec"):
            execute_level_comparison_analysis_ratio(landmarks_data, selected_feature, point1, point2, calc_type1, point3, point4, calc_type2)

    elif measurement_type == "곡률 패턴":
        st.write("#### 곡률 패턴 분석 설정")
        point_group_input = st.text_input(
            "점 번호들 (쉼표로 구분)",
            value="33,161,160,159,158",
            help="예: 33,161,160,159,158 (5개 점)",
            key="level_curvature_points"
        )

        # 점 번호들을 파싱
        try:
            point_group = [int(x.strip()) for x in point_group_input.split(',') if x.strip()]
            if len(point_group) < 3:
                st.error("최소 3개 이상의 점이 필요합니다.")
            elif len(point_group) > 10:
                st.error("최대 10개까지만 입력 가능합니다.")
            else:
                st.success(f"{len(point_group)}개 점 선택됨")

                if st.button("레벨별 곡률 패턴 분석 실행", key="level_curvature_exec"):
                    execute_level_curvature_analysis(landmarks_data, selected_feature, point_group)
        except:
            st.error("올바른 숫자 형식으로 입력하세요.")
            point_group = [33, 161, 160, 159, 158]


def render_database_management_sidebar():
    """사이드바에 데이터베이스 관리 기능 렌더링"""
    st.sidebar.write("### 🗄️ 데이터베이스 관리")

    # DB 통계 표시
    db_data = crud_service.get_dataframe()
    total_records = len(db_data)
    records_with_landmarks = len(db_data[db_data['landmarks'].notna()])

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📊 전체 데이터", total_records)
    with col2:
        st.metric("📍 Landmarks", records_with_landmarks)

    # DB 새로고침 버튼
    if st.sidebar.button("🔄 DB 새로고침",
                       help="데이터베이스에서 최신 데이터를 다시 불러옵니다.",
                       use_container_width=True):
        # 캐시 클리어 및 페이지 재실행
        st.cache_data.clear()
        st.rerun()


def render_statistical_correlation_tab():
    """통계적 연관성 분석 탭 렌더링"""
    st.header("🔬 통계적 연관성 분석")
    st.markdown("독립 변수(Atomic, Derived, Tag) vs 종속 변수(Target Tag) 간의 통계적 관계 분석")

    # session_state 초기화
    if 'independent_vars' not in st.session_state:
        st.session_state.independent_vars = []
    if 'target_tag' not in st.session_state:
        st.session_state.target_tag = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    st.divider()

    # 좌우 레이아웃
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("📥 독립 변수 선택 (최대 5개)")
        render_independent_variables_ui()

    with col_right:
        st.subheader("📤 종속 변수 (Target)")
        render_target_variable_ui()

    st.divider()

    # 독립 변수가 선택되었을 때만 분석 진행
    num_vars = len(st.session_state.independent_vars)

    if num_vars > 0 and st.session_state.target_tag:
        # 통계 기법 추천
        st.subheader("📊 적용 가능한 통계 기법")
        render_statistical_methods_ui(num_vars)

        st.divider()

        # 분석 결과
        st.subheader("📈 분석 결과")
        render_analysis_results_ui(num_vars)
    else:
        st.info("💡 좌측에서 독립 변수를, 우측에서 종속 변수를 선택하세요.")


def render_independent_variables_ui():
    """독립 변수 선택 UI"""
    # 변수 타입 옵션 로드
    atomic_options = get_available_atomic_measurements()
    derived_options = get_available_derived_measurements()
    tag_options = get_available_tags()

    # 타입 매핑 (소문자 <-> 대문자)
    type_map = {"atomic": "Atomic", "derived": "Derived", "tag": "Tag"}
    type_reverse_map = {"Atomic": "atomic", "Derived": "derived", "Tag": "tag"}

    # 현재 선택된 변수들 표시
    for idx in range(len(st.session_state.independent_vars)):
        col1, col2, col3, col4 = st.columns([2, 3, 3, 1])

        with col1:
            # 현재 저장된 타입을 대문자로 변환
            current_type = st.session_state.independent_vars[idx].get('type', 'atomic')
            current_type_display = type_map.get(current_type, "Atomic")

            var_type = st.selectbox(
                f"변수 {idx+1} 타입",
                ["Atomic", "Derived", "Tag"],
                key=f"var_type_{idx}",
                index=["Atomic", "Derived", "Tag"].index(current_type_display)
            )

            # 타입이 변경되었으면 데이터 초기화
            new_type_lowercase = type_reverse_map.get(var_type, 'atomic')
            if current_type != new_type_lowercase:
                st.session_state.independent_vars[idx] = {
                    'type': new_type_lowercase,
                    'name': None,
                    'display': None
                }

        with col2:
            if var_type == "Atomic":
                if atomic_options:
                    options = [a['display'] for a in atomic_options]
                    selected_display = st.selectbox(
                        f"값 선택",
                        options,
                        key=f"var_value_{idx}"
                    )
                    # display에서 실제 데이터 찾기
                    selected_data = next((a for a in atomic_options if a['display'] == selected_display), None)
                    if selected_data:
                        st.session_state.independent_vars[idx] = {
                            'type': 'atomic',
                            'name': selected_data['name'],
                            'side': selected_data['side'],
                            'display': selected_display
                        }
                else:
                    st.warning("Atomic measurement 데이터가 없습니다.")

            elif var_type == "Derived":
                if derived_options:
                    options = [d['display'] for d in derived_options]
                    selected_display = st.selectbox(
                        f"값 선택",
                        options,
                        key=f"var_value_{idx}"
                    )
                    selected_data = next((d for d in derived_options if d['display'] == selected_display), None)
                    if selected_data:
                        st.session_state.independent_vars[idx] = {
                            'type': 'derived',
                            'name': selected_data['name'],
                            'side': selected_data.get('side'),
                            'display': selected_display
                        }
                else:
                    st.warning("Derived measurement 데이터가 없습니다.")

            else:  # Tag
                if tag_options:
                    selected_tag = st.selectbox(
                        f"값 선택",
                        tag_options,
                        key=f"var_value_{idx}"
                    )
                    st.session_state.independent_vars[idx] = {
                        'type': 'tag',
                        'name': selected_tag,
                        'display': selected_tag
                    }
                else:
                    st.warning("Tag 데이터가 없습니다.")

        with col3:
            st.text(f"선택: {st.session_state.independent_vars[idx].get('display', 'N/A')}")

        with col4:
            if st.button("❌", key=f"remove_{idx}"):
                st.session_state.independent_vars.pop(idx)
                st.rerun()

    # 행 추가 버튼
    if len(st.session_state.independent_vars) < 5:
        if st.button("➕ 독립 변수 추가"):
            st.session_state.independent_vars.append({'type': 'atomic', 'name': None, 'display': None})
            st.rerun()
    else:
        st.warning("최대 5개까지만 추가 가능합니다.")


def render_target_variable_ui():
    """종속 변수 선택 UI"""
    tag_options = get_available_tags()

    if tag_options:
        selected_target = st.selectbox(
            "Target Tag 선택:",
            ["선택하세요"] + tag_options,
            key="target_tag_select"
        )

        if selected_target != "선택하세요":
            st.session_state.target_tag = selected_target
            st.success(f"✅ Target: **{selected_target}**")
        else:
            st.session_state.target_tag = None
    else:
        st.warning("사용 가능한 태그가 없습니다.")


def render_statistical_methods_ui(num_vars):
    """통계 기법 추천 UI"""
    recommendations = recommend_statistical_methods(num_vars)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**📊 권장 통계 기법:**")
        for method in recommendations['methods']:
            st.write(f"✓ {method}")

    with col2:
        st.write("**📈 권장 시각화:**")
        for viz in recommendations['visualizations']:
            st.write(f"✓ {viz}")


def render_analysis_results_ui(num_vars):
    """분석 결과 UI"""
    # 분석 실행 버튼
    if st.button("🔄 분석 실행", type="primary"):
        with st.spinner("분석 중..."):
            try:
                # 데이터셋 준비
                df = prepare_statistical_dataset(
                    st.session_state.independent_vars,
                    st.session_state.target_tag
                )

                if df.empty or len(df) < 10:
                    st.error("데이터가 부족합니다. 최소 10개 이상의 샘플이 필요합니다.")
                    return

                st.success(f"✅ {len(df)}개의 샘플 데이터 준비 완료")

                # 변수 이름 리스트
                var_names = [var['display'] for var in st.session_state.independent_vars]

                # 개수별 분석 실행
                if num_vars == 1:
                    # 1:1 분석
                    st.write("### 📊 Pearson 상관계수 분석")
                    pearson_result = analyze_1to1_pearson(df, var_names[0])
                    st.write(f"- 상관계수: {pearson_result['correlation']:.3f}")
                    st.write(f"- p-value: {pearson_result['p_value']:.4f}")
                    st.write(f"- 유의성: {'✅ 유의함 (p < 0.05)' if pearson_result['significant'] else '❌ 유의하지 않음'}")

                    st.write("### 📊 Spearman 상관계수 분석")
                    spearman_result = analyze_1to1_spearman(df, var_names[0])
                    st.write(f"- 상관계수: {spearman_result['correlation']:.3f}")
                    st.write(f"- p-value: {spearman_result['p_value']:.4f}")

                    st.write("### 📊 t-test 분석")
                    ttest_result = analyze_1to1_ttest(df, var_names[0])
                    st.write(f"- t-통계량: {ttest_result['t_statistic']:.3f}")
                    st.write(f"- p-value: {ttest_result['p_value']:.4f}")
                    st.write(f"- 평균 (Without Tag): {ttest_result['mean_group_0']:.3f}")
                    st.write(f"- 평균 (With Tag): {ttest_result['mean_group_1']:.3f}")

                    # 시각화
                    st.write("### 📈 시각화")
                    viz_type = st.selectbox("시각화 선택", ["Scatter Plot", "Box Plot", "Bar Chart"])
                    fig = visualize_1to1(df, var_names[0], viz_type)
                    st.plotly_chart(fig, use_container_width=True)

                elif num_vars == 2:
                    # 2:1 분석
                    st.write("### 📊 Multiple Regression 분석")
                    reg_result = analyze_2to1_regression(df, var_names)
                    st.write(f"- R² Score: {reg_result['r2_score']:.3f}")
                    st.write(f"- 계수: {reg_result['coefficients']}")
                    st.write(f"- 절편: {reg_result['intercept']:.3f}")

                    st.write("### 📊 ANOVA 분석")
                    anova_result = analyze_2to1_anova(df, var_names)
                    st.write(f"- {var_names[0]} F-통계량: {anova_result['var_0_f_stat']:.3f}, p-value: {anova_result['var_0_p_value']:.4f}")
                    st.write(f"- {var_names[1]} F-통계량: {anova_result['var_1_f_stat']:.3f}, p-value: {anova_result['var_1_p_value']:.4f}")

                    # 시각화
                    st.write("### 📈 시각화")
                    viz_type = st.selectbox("시각화 선택", ["3D Scatter", "Heatmap"])
                    fig = visualize_2to1(df, var_names, viz_type)
                    st.plotly_chart(fig, use_container_width=True)

                elif num_vars == 3:
                    # 3:1 분석
                    st.write("### 📊 Random Forest 분석")
                    rf_result = analyze_3to1_random_forest(df, var_names)
                    st.write(f"- 정확도: {rf_result['accuracy']:.3f}")
                    st.write(f"- Feature Importances:")
                    for i, importance in enumerate(rf_result['feature_importances']):
                        st.write(f"  - {var_names[i]}: {importance:.3f}")

                    # 시각화
                    st.write("### 📈 시각화")
                    viz_type = st.selectbox("시각화 선택", ["Feature Importance", "Heatmap"])
                    fig = visualize_3to1(df, var_names, viz_type, rf_result)
                    st.plotly_chart(fig, use_container_width=True)

                else:  # 4개 이상
                    # 4:1+ 분석
                    st.write("### 📊 PCA 분석")
                    pca_result = analyze_4plus_pca(df, var_names)
                    st.write(f"- 설명된 분산 비율: {pca_result['explained_variance_ratio']}")

                    st.write("### 📊 t-SNE 분석")
                    tsne_result = analyze_4plus_tsne(df, var_names)

                    # 시각화
                    st.write("### 📈 시각화")
                    viz_type = st.selectbox("시각화 선택", ["Embedding Plot", "Parallel Coordinates"])

                    if viz_type == "Embedding Plot":
                        embedding_method = st.radio("차원 축소 방법", ["PCA", "t-SNE"])
                        if embedding_method == "PCA":
                            fig = visualize_4plus(df, var_names, viz_type, pca_result)
                        else:
                            fig = visualize_4plus(df, var_names, viz_type, tsne_result)
                    else:
                        fig = visualize_4plus(df, var_names, viz_type)

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()