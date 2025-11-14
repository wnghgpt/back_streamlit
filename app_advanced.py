"""
Face Coordinate Analyzer
실시간 좌표 계산 기반 얼굴 분석 플랫폼
"""

import streamlit as st
import pandas as pd
 

# Utils modules (back_streamlit - 먼저 import해야 함!)
from utils.tag_processor import (
    analyze_tag_relationships,
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

# Database (back_analysis - utils import 이후에 경로 추가)
import sys
sys.path.insert(0, "/home/wavus/새 폴더/back_analysis/src")

from database.crud import crud_service

# Initialize db_manager (crud_service 사용)

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

    # 탭 생성 (순서: 통계적 연관성 분석 → 태그 관계도)
    tab_stat, tab_sankey = st.tabs([
        "🔬 통계적 연관성 분석",
        "🌊 태그 관계도",
    ])

    with tab_stat:
        render_statistical_correlation_tab()

    with tab_sankey:
        render_sankey_diagram_tab(landmarks_data)


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




def render_database_management_sidebar():
    """사이드바에 데이터베이스 관리 기능 렌더링"""
    st.sidebar.write("### 🗄️ 데이터베이스 관리")

    # DB 통계 표시
    db_data = crud_service.get_dataframe()
    total_records = len(db_data)
    records_with_landmarks = len(db_data[db_data['landmarks'].notna()]) if 'landmarks' in db_data.columns else 0

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

    # session_state 초기화
    if 'independent_vars' not in st.session_state:
        # 초기 로딩 시 독립변수 선택 드롭박스 1개 기본 제공
        st.session_state.independent_vars = [
            {'type': 'atomic', 'name': None, 'display': None}
        ]
    if 'target_tag' not in st.session_state:
        st.session_state.target_tag = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

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

    # 최소 1개 항목 보장 (사용자가 모두 제거한 경우에도)
    if len(st.session_state.independent_vars) == 0:
        st.session_state.independent_vars.append({'type': 'atomic', 'name': None, 'display': None})

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
                    for viz_type in ["Scatter Plot", "Box Plot", "Bar Chart"]:
                        st.write(f"#### {viz_type}")
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
                    for viz_type in ["3D Scatter", "Heatmap"]:
                        st.write(f"#### {viz_type}")
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
                    for viz_type in ["Feature Importance", "Heatmap"]:
                        st.write(f"#### {viz_type}")
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

                    # Embedding Plot - PCA
                    st.write("#### Embedding Plot (PCA)")
                    fig = visualize_4plus(df, var_names, "Embedding Plot", pca_result)
                    st.plotly_chart(fig, use_container_width=True)

                    # Embedding Plot - t-SNE
                    st.write("#### Embedding Plot (t-SNE)")
                    fig = visualize_4plus(df, var_names, "Embedding Plot", tsne_result)
                    st.plotly_chart(fig, use_container_width=True)

                    # Parallel Coordinates
                    st.write("#### Parallel Coordinates")
                    fig = visualize_4plus(df, var_names, "Parallel Coordinates")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
