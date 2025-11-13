"""
프로필 관리 유틸리티
ReferenceProfile CRUD 기능
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys

# back_analysis import
sys.path.insert(0, "/home/wavus/새 폴더/back_analysis/src")
from database.connection import DatabaseManager
from database.models import ReferenceProfile


def get_all_profiles_with_images(sort_by="최신순"):
    """이미지가 있는 프로필 조회 (정렬 옵션)"""
    db_manager = DatabaseManager()

    with db_manager.get_session() as session:
        query = session.query(ReferenceProfile)\
            .filter(ReferenceProfile.image_file_path.isnot(None))

        # 정렬 옵션 적용
        if sort_by == "최신순":
            query = query.order_by(ReferenceProfile.upload_date.desc())
        elif sort_by == "오래된순":
            query = query.order_by(ReferenceProfile.upload_date.asc())
        elif sort_by == "이름순":
            query = query.order_by(ReferenceProfile.name.asc())
        elif sort_by == "ID순":
            query = query.order_by(ReferenceProfile.id.desc())

        profiles = query.all()

        # 세션이 닫히기 전에 딕셔너리로 변환
        result = []
        for profile in profiles:
            result.append({
                'id': profile.id,
                'name': profile.name,
                'image_file_path': profile.image_file_path,
                'json_file_path': profile.json_file_path,
                'upload_date': profile.upload_date,
                'landmarks_json': profile.landmarks_json,
                'ratios_json': profile.ratios_json
            })

        return result


def get_profile_by_id(profile_id):
    """ID로 프로필 조회"""
    db_manager = DatabaseManager()

    with db_manager.get_session() as session:
        profile = session.query(ReferenceProfile).filter_by(id=profile_id).first()

        if not profile:
            return None

        # 관련 데이터 카운트
        tags_count = len(profile.tags) if profile.tags else 0
        landmarks_count = len(profile.landmarks_points) if profile.landmarks_points else 0
        ratios_count = len(profile.basic_ratio) if profile.basic_ratio else 0

        return {
            'id': profile.id,
            'name': profile.name,
            'full_name': profile.full_name if hasattr(profile, 'full_name') else profile.name,
            'last_name': profile.last_name if hasattr(profile, 'last_name') else "",
            'first_name': profile.first_name if hasattr(profile, 'first_name') else "",
            'romanized_name': profile.romanized_name if hasattr(profile, 'romanized_name') else "",
            'image_file_path': profile.image_file_path,
            'json_file_path': profile.json_file_path,
            'upload_date': profile.upload_date,
            'landmarks_json': profile.landmarks_json,
            'ratios_json': profile.ratios_json,
            'tags_count': tags_count,
            'landmarks_count': landmarks_count,
            'ratios_count': ratios_count
        }


def update_profile(profile_id, name=None, json_file_path=None, image_file_path=None):
    """프로필 정보 업데이트 (이름 변경 시 자동 파싱)"""
    db_manager = DatabaseManager()

    with db_manager.get_session() as session:
        profile = session.query(ReferenceProfile).filter_by(id=profile_id).first()

        if not profile:
            return {"success": False, "message": "프로필을 찾을 수 없습니다."}

        # 업데이트
        if name is not None:
            # 한글 이름 파싱
            sys.path.insert(0, "/home/wavus/새 폴더/back_analysis/src")
            from utils.korean_name_parser import parse_korean_name, romanize_korean_name

            full_name, last_name, first_name = parse_korean_name(name)
            romanized = romanize_korean_name(name)

            if not full_name:
                full_name = name
                last_name = ""
                first_name = ""

            profile.name = full_name
            profile.full_name = full_name
            profile.last_name = last_name
            profile.first_name = first_name
            profile.romanized_name = romanized

        if json_file_path is not None:
            profile.json_file_path = json_file_path
        if image_file_path is not None:
            profile.image_file_path = image_file_path

        session.commit()

        return {"success": True, "message": "프로필이 업데이트되었습니다."}


def delete_profiles(profile_ids):
    """프로필 삭제 (cascade로 관련 데이터도 삭제)"""
    db_manager = DatabaseManager()

    with db_manager.get_session() as session:
        deleted_count = 0

        for profile_id in profile_ids:
            profile = session.query(ReferenceProfile).filter_by(id=profile_id).first()
            if profile:
                session.delete(profile)
                deleted_count += 1

        session.commit()

        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"{deleted_count}개 프로필이 삭제되었습니다."
        }


def get_image_path(image_file_path):
    """이미지 경로를 절대 경로로 변환"""
    if not image_file_path:
        return None

    # /uploads/... 형태면 앞의 / 제거
    if image_file_path.startswith('/uploads/'):
        image_file_path = image_file_path[1:]

    # 상대 경로면 절대 경로로 변환
    if not os.path.isabs(image_file_path):
        image_file_path = f"/home/wavus/새 폴더/back_analysis/{image_file_path}"

    return image_file_path


@st.dialog("프로필 상세 정보", width="large")
def show_profile_modal(profile_id):
    """프로필 상세 정보 모달"""
    profile = get_profile_by_id(profile_id)

    if not profile:
        st.error("프로필을 찾을 수 없습니다.")
        return

    # 이미지 표시
    image_path = get_image_path(profile['image_file_path'])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🖼️ 이미지")
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"이미지 로드 실패: {e}")
        else:
            st.warning("이미지 없음")

    with col2:
        st.markdown("### 📋 기본 정보")

        # 편집 가능한 필드
        new_name = st.text_input("전체 이름", value=profile.get('full_name') or profile['name'], key=f"edit_name_{profile_id}")

        # 성/이름/로마자 표시 (읽기 전용)
        if profile.get('last_name') or profile.get('first_name'):
            col_a, col_b = st.columns(2)
            with col_a:
                st.text_input("성", value=profile.get('last_name', ''), key=f"view_last_{profile_id}", disabled=True)
            with col_b:
                st.text_input("이름", value=profile.get('first_name', ''), key=f"view_first_{profile_id}", disabled=True)
            st.caption("*성/이름은 자동 파싱되며 직접 수정할 수 없습니다.")

        if profile.get('romanized_name'):
            st.text_input("로마자 표기 (Romanized)", value=profile.get('romanized_name', ''), key=f"view_romanized_{profile_id}", disabled=True)
            st.caption("*파일명에 사용: processed_{romanized}_{uuid}.jpg")

        new_json_path = st.text_input("JSON 경로", value=profile['json_file_path'] or "", key=f"edit_json_{profile_id}")
        new_image_path = st.text_input("이미지 경로", value=profile['image_file_path'] or "", key=f"edit_image_{profile_id}")

        st.divider()

        # 읽기 전용 정보
        st.markdown(f"**ID:** `{profile['id']}`")
        st.markdown(f"**업로드 날짜:** {profile['upload_date'].strftime('%Y-%m-%d %H:%M:%S') if profile['upload_date'] else 'N/A'}")

        st.divider()

        st.markdown("### 📊 관련 데이터")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("태그", profile['tags_count'])
        with col_b:
            st.metric("랜드마크", profile['landmarks_count'])
        with col_c:
            st.metric("비율", profile['ratios_count'])

    st.divider()

    # 저장 버튼
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("💾 저장", type="primary", use_container_width=True):
            result = update_profile(
                profile_id,
                name=new_name,
                json_file_path=new_json_path if new_json_path else None,
                image_file_path=new_image_path if new_image_path else None
            )

            if result['success']:
                st.success(result['message'])
                st.rerun()
            else:
                st.error(result['message'])


def render_profile_management_ui():
    """프로필 관리 UI 렌더링"""

    # 1. 정렬 옵션 및 통계
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        sort_by = st.selectbox(
            "🔽 정렬:",
            ["최신순", "오래된순", "이름순", "ID순"]
        )

    # 2. DB에서 모든 프로필 조회
    all_profiles = get_all_profiles_with_images(sort_by)

    if not all_profiles:
        st.warning("⚠️ 이미지가 있는 프로필이 없습니다.")
        return

    with col2:
        st.metric("전체 프로필", f"{len(all_profiles)}개")

    # 3. 삭제할 프로필 추적 (세션 상태)
    if 'profiles_to_delete' not in st.session_state:
        st.session_state.profiles_to_delete = set()

    with col3:
        st.metric("삭제 예정", f"{len(st.session_state.profiles_to_delete)}개")

    st.divider()

    # 4. 페이지네이션 설정
    page_size = 36  # 6행 × 6열
    total_pages = (len(all_profiles) + page_size - 1) // page_size

    # 현재 페이지
    page = int(st.session_state.get("profile_page_bottom", 1))
    if total_pages <= 0:
        page = 1
        start_idx, end_idx = 0, 0
    else:
        if page < 1:
            page = 1
        if page > total_pages:
            page = total_pages
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(all_profiles))

    # 5. 이미지 그리드 (6열)
    st.markdown("### 🖼️ 프로필 관리")

    # 6개씩 행으로 묶기
    for row_start in range(start_idx, end_idx, 6):
        cols = st.columns(6)

        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= end_idx:
                break

            profile = all_profiles[idx]

            with col:
                # 이미지 표시
                image_path = get_image_path(profile['image_file_path'])

                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        # 이미지 클릭 시 모달 (버튼으로 구현)
                        if st.button(
                            "🔍",
                            key=f"view_{profile['id']}",
                            use_container_width=True,
                            help="클릭하여 상세보기"
                        ):
                            show_profile_modal(profile['id'])

                        st.image(image, use_container_width=True)
                    except Exception as e:
                        st.error(f"이미지 로드 실패")
                else:
                    st.warning("이미지 없음")

                # 프로필 정보
                st.markdown(f"**{profile['name']}**")
                st.caption(f"ID: {profile['id']}")

                # 삭제 체크박스
                is_checked = st.checkbox(
                    "🗑️ 삭제",
                    value=(profile['id'] in st.session_state.profiles_to_delete),
                    key=f"delete_{profile['id']}"
                )

                # 체크 상태 업데이트
                if is_checked:
                    st.session_state.profiles_to_delete.add(profile['id'])
                else:
                    st.session_state.profiles_to_delete.discard(profile['id'])

    st.divider()

    # 6. 페이지네이션 (하단)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"페이지 (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            key="profile_page_bottom"
        )

    st.divider()

    # 7. 확정 버튼
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("✅ 확정 및 저장", type="primary", use_container_width=True):
            if len(st.session_state.profiles_to_delete) == 0:
                st.info("변경사항이 없습니다.")
            else:
                # 삭제 실행
                with st.spinner("삭제 중..."):
                    result = delete_profiles(list(st.session_state.profiles_to_delete))

                if result['success']:
                    st.success(f"✅ {result['deleted_count']}개 프로필이 삭제되었습니다!")

                    # 세션 상태 초기화
                    st.session_state.profiles_to_delete = set()

                    st.balloons()

                    # 2초 후 자동 새로고침
                    import time
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"삭제 실패: {result.get('message', '알 수 없는 오류')}")
