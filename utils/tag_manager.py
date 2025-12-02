"""
태그 관리 유틸리티
이미지 기반 수동 태그 라벨링
"""
import streamlit as st
import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys
from pathlib import Path

# back_analysis 경로 동적 추가
ROOT_DIR = Path(__file__).resolve().parent.parent
BACK_ANALYSIS_SRC = ROOT_DIR.parent / "back_analysis" / "src"
if BACK_ANALYSIS_SRC.exists():
    sys.path.insert(0, str(BACK_ANALYSIS_SRC))
from database.connection import DatabaseManager
from database.models import ReferenceProfile, ReferenceTag, TagDefinition
from database.crud import crud_service

# utils import
from .tag_processor import get_tag_groups
import re


def _load_level_lookup():
    """tag_classification.json 기반 레벨별 태그 집합을 로드"""
    return {level: set(tags) for level, tags in get_tag_groups().items()}


def _get_level_for_tag(tag_name: str) -> int:
    """태그가 속한 숫자 레벨(1/2/3) 반환. 없으면 기본 2."""
    levels = _load_level_lookup()
    for level, tags in levels.items():
        if tag_name in tags:
            return int(level)
    return 2


def get_fs_level(tag_name: str) -> int:
    """파일/폴더 레벨은 태그 레벨과 동일하게 사용"""
    return _get_level_for_tag(tag_name)


def get_db_level(tag_name: str) -> int:
    """DB 저장용 레벨(ReferenceTag.tag_level)"""
    return _get_level_for_tag(tag_name)


def safe_tag_filename(tag: str) -> str:
    """태그명을 파일명으로 안전하게 변환"""
    if not isinstance(tag, str):
        tag = str(tag)
    s = tag.strip().replace('/', '_').replace('\\', '_')
    # 한글, 영문, 숫자, 공백, 점, 대시, 밑줄만 허용
    return re.sub(r"[^\w\-. \uAC00-\uD7A3]", '_', s)


def get_all_available_tags():
    """모든 사용 가능한 태그 목록 반환"""
    tag_groups = get_tag_groups()
    all_tags = []
    for tags in tag_groups.values():
        all_tags.extend(tags)
    return sorted(all_tags)


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
            query = query.order_by(ReferenceProfile.full_name.asc())
        elif sort_by == "ID순":
            query = query.order_by(ReferenceProfile.id.desc())

        profiles = query.all()

        # 세션이 닫히기 전에 딕셔너리로 변환
        result = []
        for profile in profiles:
            result.append({
                'id': profile.id,
                'name': getattr(profile, 'full_name', None) or getattr(profile, 'name', ''),
                'image_file_path': profile.image_file_path,
                'upload_date': profile.upload_date
            })

        return result


def load_tag_annotation(tag_name):
    """태그 annotation JSON 파일 로드 (source_data/tags 기준)"""
    fs_level = get_fs_level(tag_name)
    filename = safe_tag_filename(tag_name) + ".json"
    base_dir = BACK_ANALYSIS_SRC / "database" / "source_data" / "tags"
    json_path = base_dir / f"level_{fs_level}" / filename

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            tag_data = json.load(f)

        profile_ids = set()
        if 'profiles' in tag_data:
            for profile in tag_data['profiles']:
                if isinstance(profile, dict) and 'id' in profile:
                    profile_ids.add(profile['id'])
                elif isinstance(profile, str):
                    pass
        return profile_ids
    else:
        return set()


def save_tag_annotation(tag_name, selected_profiles):
    """태그 annotation JSON 파일 저장"""
    fs_level = get_fs_level(tag_name)
    db_level = get_db_level(tag_name)
    json_dir = BACK_ANALYSIS_SRC / "database" / "source_data" / "tags" / f"level_{fs_level}"
    json_path = json_dir / f"{safe_tag_filename(tag_name)}.json"

    # 디렉토리 생성
    json_dir.mkdir(parents=True, exist_ok=True)

    # JSON 데이터 구성
    tag_data = {
        "tag_name": tag_name,
        # DB 의미의 레벨 저장 (0:추상,1:1차,2:2차)
        "tag_level": db_level,
        "description": "",
        "profiles": selected_profiles,
        "last_updated": datetime.now().isoformat()
    }

    # 파일 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tag_data, f, ensure_ascii=False, indent=2)

    return json_path


def sync_json_to_db(json_path):
    """JSON 파일 → DB 동기화 (수동)"""
    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        tag_data = json.load(f)

    tag_name = tag_data['tag_name']
    # DB 저장용 레벨은 JSON의 값을 신뢰하지 않고 재계산
    tag_level = get_db_level(tag_name)
    canonical_name = tag_name
    tag_value = None
    if tag_level == 1 and '-' in tag_name:
        parts = tag_name.split('-')
        if len(parts) >= 2:
            canonical_name = '-'.join(parts[:-1])
            tag_value = parts[-1]
    profiles = tag_data.get('profiles', [])

    # profiles에서 id 추출
    profile_id_set = set()
    for profile in profiles:
        if isinstance(profile, dict) and 'id' in profile:
            profile_id_set.add(profile['id'])

    db_manager = DatabaseManager()

    with db_manager.get_session() as session:
        # DB에서 모든 프로필 조회 (이름 매핑용)
        all_profiles = session.query(ReferenceProfile).all()
        id_to_name = {p.id: (getattr(p, 'full_name', None) or getattr(p, 'name', str(p.id))) for p in all_profiles}

        added_count = 0
        removed_count = 0

        for profile in all_profiles:
            # DB의 현재 태그 확인
            existing_tag = session.query(ReferenceTag).filter_by(
                profile_id=profile.id,
                tag_name=canonical_name,
                tag_value=tag_value
            ).first()

            should_have_tag = profile.id in profile_id_set

            if should_have_tag and not existing_tag:
                # 태그 추가
                new_tag = ReferenceTag(
                    profile_id=profile.id,
                    tag_name=canonical_name,
                    tag_level=tag_level,
                    tag_value=tag_value
                )
                session.add(new_tag)
                added_count += 1

            elif not should_have_tag and existing_tag:
                # 태그 제거
                session.delete(existing_tag)
                removed_count += 1

        # TagDefinition upsert (JSON 기반 요약 정보)
        profile_ids_sorted = sorted(list(profile_id_set))
        # 안전장치: JSON에 프로필 목록이 비어있으면 DB 기준으로 역산
        if not profile_ids_sorted:
            db_ids = [rt.profile_id for rt in session.query(ReferenceTag).filter_by(tag_name=tag_name).all()]
            profile_ids_sorted = sorted(list(set(db_ids)))
        profile_names_sorted = [id_to_name.get(pid, str(pid)) for pid in profile_ids_sorted]

        tag_def = session.query(TagDefinition).filter_by(tag_name=tag_name).first()
        if tag_def:
            tag_def.tag_level = tag_level
            tag_def.description = tag_data.get('description') or ""
            tag_def.profile_ids = profile_ids_sorted
            tag_def.profile_names = profile_names_sorted
            tag_def.profile_count = len(profile_ids_sorted)
            tag_def.source_file = Path(json_path).name
        else:
            tag_def = TagDefinition(
                tag_name=tag_name,
                tag_level=tag_level,
                description=tag_data.get('description') or "",
                profile_ids=profile_ids_sorted,
                profile_names=profile_names_sorted,
                profile_count=len(profile_ids_sorted),
                source_file=Path(json_path).name
            )
            session.add(tag_def)

        session.commit()

    return {
        "added": added_count,
        "removed": removed_count,
        "total": len(profile_id_set)
    }


def render_tag_management_ui():
    """태그 관리 UI 렌더링"""

    # 1. 헤더 부제목 제거 (요청 반영)

    # 2. 레벨 선택 및 태그 선택 및 정렬 옵션
    tag_groups = get_tag_groups()
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.8])

    with col1:
        selected_level = st.selectbox(
            "📊 레벨:",
            sorted(tag_groups.keys()),
            format_func=lambda x: f"{x}차",
            help="tag_classification.json의 레벨(1/2/3)을 그대로 사용합니다."
        )

    # 레벨에 따라 태그 필터링
    filtered_tags = []

    if selected_level == 1:
        # 태그명을 "부위-속성 / 값" 두 단계로 분리
        level1_tags = tag_groups.get(1, [])
        base_to_values = {}
        for t in level1_tags:
            parts = t.split('-')
            if len(parts) >= 2:
                base = '-'.join(parts[:-1])  # 부위-속성
                value = parts[-1]            # 값
            else:
                base = t
                value = None
            base_to_values.setdefault(base, set())
            if value:
                base_to_values[base].add(value)

        if not base_to_values:
            st.warning("1차(레벨1) 태그가 없습니다.")
            return

        with col2:
            selected_base = st.selectbox(
                "📌 태그 유형",
                sorted(base_to_values.keys()),
                help="부위-속성을 먼저 선택하세요 (예: eye-길이)"
            )
        with col3:
            values = sorted(base_to_values.get(selected_base, []))
            if values:
                selected_value = st.selectbox(
                    "태그 값",
                    values,
                    help="세부 값을 선택하세요 (예: 긴)"
                )
                selected_tag = f"{selected_base}-{selected_value}"
            else:
                selected_tag = selected_base
    else:
        if selected_level == 2:
            filtered_tags.extend(tag_groups.get(2, []))
        elif selected_level == 3:
            filtered_tags.extend(tag_groups.get(3, []))

        with col2:
            if filtered_tags:
                selected_tag = st.selectbox(
                    "📌 태그 선택:",
                    sorted(filtered_tags),
                    help="분석할 태그를 선택하세요"
                )
            else:
                st.warning(f"{selected_level}차 태그가 없습니다.")
                return

    with col4:
        sort_by = st.selectbox(
            "🔽 정렬:",
            ["최신순", "오래된순", "이름순", "ID순"]
        )

    # 3. 현재 JSON 파일 로드
    current_profile_ids = load_tag_annotation(selected_tag)

    with col5:
        st.markdown(f"**선택**: {len(current_profile_ids)}개")

    # 4. DB에서 모든 프로필 조회
    all_profiles = get_all_profiles_with_images(sort_by)

    if not all_profiles:
        st.warning("⚠️ 이미지가 있는 프로필이 없습니다.")
        return

    st.divider()

    # 5. 페이지네이션 설정 (이미지 표시 전)
    page_size = 36  # 6행 × 6열
    total_pages = (len(all_profiles) + page_size - 1) // page_size

    # 현재 페이지 결정 및 인덱스 계산
    # 하단 페이지 입력값이 세션 상태에 저장되므로 우선 사용
    page = int(st.session_state.get("page_bottom", 1))
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

    # 6. 이미지 그리드 (6열)
    st.markdown("### 🖼️ 프로필 선택")

    # 프로필 id->name 매핑 (저장 시 사용)
    id_to_name = {p['id']: p['name'] for p in all_profiles}

    checkbox_states = {}

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
                image_path = profile['image_file_path']

                # back_analysis/uploads/ 경로 처리 (repo 루트 기준)
                if image_path:
                    if image_path.startswith('/uploads/'):
                        image_path = image_path[1:]  # /uploads/ -> uploads/
                    if not os.path.isabs(image_path):
                        base_dir = Path(__file__).resolve().parent.parent.parent / "back_analysis"
                        image_path = base_dir / image_path
                    image_path = str(image_path)

                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, use_container_width=True)
                    except Exception as e:
                        st.error(f"이미지 로드 실패: {e}")
                else:
                    st.warning("이미지 없음")

                # 체크박스
                is_checked = st.checkbox(
                    f"**{profile['name']}**\n`ID:{profile['id']}`",
                    value=(profile['id'] in current_profile_ids),
                    key=f"check_{selected_tag}_{profile['id']}"
                )

                checkbox_states[profile['id']] = {
                    'checked': is_checked,
                    'name': profile['name']
                }

    st.divider()

    # 7. 페이지네이션 (아래로 이동)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"페이지 (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            key="page_bottom"
        )

    st.divider()

    # 8. 완료 버튼
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("✅ 완료 및 저장", type="primary", use_container_width=True):
            # 현재 페이지 id 집합
            page_ids = set()
            for idx in range(start_idx, end_idx):
                page_ids.add(all_profiles[idx]['id'])

            # 체크된 id 집합(현재 페이지 기준)
            selected_ids = {pid for pid, data in checkbox_states.items() if data['checked']}

            # 기존 선택(다른 페이지) + 현재 페이지 선택
            final_ids = (set(current_profile_ids) - page_ids) | selected_ids

            # 저장용 프로필 목록 구성
            selected_profiles = [
                {"id": pid, "name": id_to_name.get(pid, str(pid))}
                for pid in sorted(final_ids)
            ]

            # JSON 저장
            json_path = save_tag_annotation(selected_tag, selected_profiles)

            # DB 동기화
            with st.spinner("DB 동기화 중..."):
                result = sync_json_to_db(json_path)

            # 결과 표시
            st.success(f"✅ **{selected_tag}** 태그 저장 완료!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 선택", result['total'])
            with col2:
                st.metric("추가됨", result['added'], delta=result['added'] if result['added'] > 0 else None)
            with col3:
                st.metric("제거됨", result['removed'], delta=-result['removed'] if result['removed'] > 0 else None)

            st.balloons()

            # 3초 후 자동 새로고침
            import time
            time.sleep(2)
            st.rerun()
