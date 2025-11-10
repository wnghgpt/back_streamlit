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
sys.path.insert(0, "/home/web_app/back_analysis/src")

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

# Page config
st.set_page_config(
    page_title="Face Coordinate Analyzer",
    page_icon="🎭",
    layout="wide"
)


def main():
    st.title("🎭 Face Coordinate Analyzer")
    st.markdown("**실시간 좌표 계산 기반 얼굴 분석 플랫폼**")

    # 사이드바에 데이터베이스 관리 기능 추가
    render_database_management_sidebar()

    # 랜드마크 데이터 로드
    landmarks_data = load_landmarks_data()

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["🧮 좌표 분석", "🔗 태그 연관성 분석", "🌊 태그 관계도", "📊 태그-수치 분석"])

    with tab1:
        render_landmarks_analysis_tab(landmarks_data)

    with tab2:
        render_tag_analysis_tab(landmarks_data)

    with tab3:
        render_sankey_diagram_tab(landmarks_data)

    with tab4:
        render_tag_analysis_tab_new(landmarks_data)


def load_landmarks_data():
    """랜드마크 데이터 로드"""
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

    # JSON 파일에서 추가 데이터 로드 및 병합
    people_json_path = Path("source_data/people_json")
    json_data_list = []
    if people_json_path.exists():
        for file_path in people_json_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # 'landmarks' 데이터가 문자열이면 파싱
                    if isinstance(json_data.get('landmarks'), str):
                        json_data['landmarks'] = json.loads(json_data['landmarks'])
                    json_data_list.append(json_data)
            except Exception as e:
                st.error(f"'{file_path.name}' 파일 로딩 오류: {e}")

    json_df = pd.DataFrame(json_data_list)

    # 3. 데이터 병합
    if not json_df.empty:
        # DB 데이터와 JSON 데이터를 합치고, 'name'을 기준으로 중복 제거 (JSON 파일 우선)
        combined_data = pd.concat([landmarks_data, json_df], ignore_index=True)
        combined_data.drop_duplicates(subset=['name'], keep='last', inplace=True)
        landmarks_data = combined_data

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

    # JSON 파일 스캔
    people_json_path = Path("source_data/people_json")
    if people_json_path.exists():
        json_files = list(people_json_path.glob("*.json"))

        if json_files:
            st.sidebar.write(f"📁 `source_data/people_json/`에서 {len(json_files)}개 파일 발견")

            # 미리보기
            with st.sidebar.expander("파일 목록 보기"):
                for file_path in json_files[:5]:  # 최대 5개만 표시
                    st.write(f"• {file_path.name}")
                if len(json_files) > 5:
                    st.write(f"... 외 {len(json_files) - 5}개")

            # 데이터베이스 추가 버튼
            if st.sidebar.button("🔄 폴더-DB 동기화",
                               help="source_data/people_json/ 폴더와 데이터베이스를 완전히 동기화합니다. (추가/수정/삭제 자동 처리)"):

                with st.spinner("폴더와 DB 동기화 중..."):
                    try:
                        # 새로운 동기화 시스템 사용
                        sync_result = db_manager.sync_with_folder("source_data/people_json")

                        if "error" in sync_result:
                            st.sidebar.error(sync_result["error"])
                        else:
                            # 결과 표시
                            st.sidebar.success(f"🔄 동기화 완료!")

                            col1, col2 = st.sidebar.columns(2)
                            with col1:
                                st.metric("➕ 추가", sync_result["added"], delta=sync_result["added"] if sync_result["added"] > 0 else None)
                                st.metric("✏️ 수정", sync_result["updated"], delta=sync_result["updated"] if sync_result["updated"] > 0 else None)
                            with col2:
                                st.metric("🗑️ 삭제", sync_result["deleted"], delta=-sync_result["deleted"] if sync_result["deleted"] > 0 else None)
                                st.metric("📁 총 파일", sync_result["total_files"])

                            if sync_result["added"] + sync_result["updated"] + sync_result["deleted"] == 0:
                                st.sidebar.info("📌 모든 데이터가 이미 동기화되어 있습니다.")
                            else:
                                st.sidebar.info("✨ source_data/people_json 폴더와 DB가 완전히 동기화되었습니다!")

                    except Exception as e:
                        st.sidebar.error(f"동기화 중 오류 발생: {e}")
        else:
            st.sidebar.info("📭 `source_data/people_json/` 폴더가 비어있습니다.")
    else:
        st.sidebar.info("📁 `source_data/people_json/` 폴더가 없습니다.")


if __name__ == "__main__":
    main()