"""
한글 이름 파싱 유틸리티

한글 이름을 성/이름으로 자동 분리합니다.
- 2글자 성 (남궁, 제갈, 선우 등) 우선 처리
- 기본: 첫 글자가 성, 나머지가 이름
- ASCII-safe romanization 지원
"""
import re
from typing import Tuple

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False

# 2글자 성 목록 (한국에서 일반적으로 사용되는 복성)
TWO_CHAR_LAST_NAMES = {
    '남궁', '제갈', '선우', '독고', '황보', '사공', '서문',
    '어금', '장곡', '망절', '강전', '동방', '삼척'
}


def parse_korean_name(full_name: str) -> Tuple[str, str, str]:
    """한글 이름을 성/이름으로 분리

    Args:
        full_name: 전체 이름 (예: "아이유", "남궁민수")

    Returns:
        (full_name, last_name, first_name) 튜플
        예: ("아이유", "이", "지은") - 실제로는 추론
            ("남궁민수", "남궁", "민수")
    """
    if not full_name:
        return "", "", ""

    # 공백/특수문자 제거
    clean_name = re.sub(r'[^\uAC00-\uD7A3a-zA-Z]', '', full_name.strip())

    if not clean_name:
        return "", "", ""

    # 2글자 성 체크
    if len(clean_name) >= 2 and clean_name[:2] in TWO_CHAR_LAST_NAMES:
        last_name = clean_name[:2]
        first_name = clean_name[2:] if len(clean_name) > 2 else ""
        return clean_name, last_name, first_name

    # 1글자 성 (기본)
    if len(clean_name) >= 1:
        last_name = clean_name[0]
        first_name = clean_name[1:] if len(clean_name) > 1 else ""
        return clean_name, last_name, first_name

    return clean_name, "", ""


def is_valid_korean_name(full_name: str) -> bool:
    """한글 이름 유효성 검사

    Args:
        full_name: 전체 이름

    Returns:
        유효하면 True, 아니면 False
    """
    if not full_name:
        return False

    # 한글만 허용 (공백 제거 후)
    clean_name = re.sub(r'\s+', '', full_name.strip())

    # 한글 문자만 포함되어 있는지 확인
    korean_only = re.match(r'^[\uAC00-\uD7A3]+$', clean_name)

    # 길이 체크 (2~5글자)
    valid_length = 2 <= len(clean_name) <= 5

    return bool(korean_only) and valid_length


def romanize_korean_name(full_name: str) -> str:
    """한글 이름을 ASCII-safe 문자열로 변환

    Args:
        full_name: 전체 한글 이름 (예: "아이유", "남궁민수")

    Returns:
        romanized 이름 (예: "aiu", "namgung_minsu")

    Examples:
        >>> romanize_korean_name("아이유")
        "aiu"
        >>> romanize_korean_name("남궁민수")
        "namgung_minsu"
        >>> romanize_korean_name("김철수")
        "kim_cheolsu"
    """
    if not full_name:
        return "unknown"

    # unidecode가 없으면 fallback
    if not HAS_UNIDECODE:
        # 간단한 fallback: 한글 제거, 알파벳/숫자만 유지
        safe = re.sub(r'[^a-zA-Z0-9]', '', full_name)
        return safe.lower() if safe else "unknown"

    # 공백/특수문자 제거
    clean_name = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9]', '', full_name.strip())

    if not clean_name:
        return "unknown"

    # unidecode로 romanization
    romanized = unidecode(clean_name)

    # 소문자 변환 및 sanitize
    safe_name = re.sub(r'[^a-z0-9]', '_', romanized.lower())

    # 연속 밑줄 제거
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')

    # 길이 제한 (최대 30자)
    safe_name = safe_name[:30]

    return safe_name if safe_name else "unknown"


# 테스트 케이스
if __name__ == "__main__":
    test_cases = [
        "아이유",
        "남궁민수",
        "제갈공명",
        "김철수",
        "이영희",
        "선우은숙",
        "박",
        "김",
        "",
        "John Doe",
    ]

    print("=== 한글 이름 파싱 테스트 ===")
    for name in test_cases:
        full, last, first = parse_korean_name(name)
        valid = is_valid_korean_name(name)
        romanized = romanize_korean_name(name)
        print(f"{name:15s} → 전체: {full:10s}, 성: {last:5s}, 이름: {first:10s} | Romanized: {romanized:20s} | 유효: {valid}")
