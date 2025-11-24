"""
태그 관리 페이지
이미지 기반 수동 태그 라벨링
"""
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가 (가장 먼저)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# back_streamlit utils import (충돌 방지를 위해 먼저 import)
from utils.tag_manager import render_tag_management_ui

# back_analysis import (동적 경로 추가)
BACK_ANALYSIS_SRC = project_root.parent / "back_analysis" / "src"
if BACK_ANALYSIS_SRC.exists():
    sys.path.insert(0, str(BACK_ANALYSIS_SRC))
from database.connection import DatabaseManager
from database.crud import crud_service

# 페이지 설정
st.set_page_config(
    page_title="태그 관리",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🏷️ 태그 관리 시스템")

st.divider()

# 사이드바 - DB 정보
with st.sidebar:
    st.markdown("### 📊 데이터베이스 정보")

    try:
        db_data = crud_service.get_dataframe()
        total_records = len(db_data)
        records_with_landmarks = len(db_data[db_data['landmarks'].notna()])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("전체", total_records)
        with col2:
            st.metric("Landmarks", records_with_landmarks)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")

    st.divider()

    # 새로고침 (좌측 패널)
    if st.button("🔄 새로고침", use_container_width=True, help="페이지를 다시 로드합니다"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

    st.markdown("### 💡 사용 방법")
    st.markdown("""
    1. **태그 선택**: 작업할 태그 선택
    2. **프로필 선택**: 해당하는 이미지 체크
    3. **완료 버튼**: 저장 및 DB 동기화
    4. **새로고침**: 최신 데이터 반영
    """)

    st.divider()

    st.markdown("### 📁 저장 위치")
    st.code("back_analysis/src/database/\ndefinitions/tags/level_X/\n{태그명}.json")

# 메인 UI 렌더링
try:
    render_tag_management_ui()
except Exception as e:
    st.error(f"오류 발생: {e}")
    st.exception(e)
