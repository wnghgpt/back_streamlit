"""
프로필 관리 페이지
ReferenceProfile CRUD
"""
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# back_analysis import
sys.path.insert(0, "/home/wavus/새 폴더/back_analysis/src")
from database.connection import DatabaseManager
from database.crud import crud_service

# utils import
from utils.profile_manager import render_profile_management_ui

# 페이지 설정
st.set_page_config(
    page_title="프로필 관리",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("👤 프로필 관리 시스템")

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

    # 새로고침
    if st.button("🔄 새로고침", use_container_width=True, help="페이지를 다시 로드합니다"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # 세션 상태 초기화
        if 'profiles_to_delete' in st.session_state:
            st.session_state.profiles_to_delete = set()
        st.rerun()

    st.markdown("### 💡 사용 방법")
    st.markdown("""
    1. **섬네일 클릭**: 상세 정보 확인 및 수정
    2. **삭제 체크**: 삭제할 프로필 선택
    3. **확정 버튼**: DB에 변경사항 반영
    4. **새로고침**: 최신 데이터 반영
    """)

    st.divider()

    st.markdown("### ⚠️ 주의사항")
    st.warning("""
    - 프로필 삭제 시 관련된 모든 데이터(태그, 랜드마크, 비율 등)가 함께 삭제됩니다.
    - 삭제된 데이터는 복구할 수 없습니다.
    """)

# 메인 UI 렌더링
try:
    render_profile_management_ui()
except Exception as e:
    st.error(f"오류 발생: {e}")
    st.exception(e)
