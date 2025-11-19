"""
통계적 연관성 분석 유틸리티
독립변수(atomic, derived, tag) vs 종속변수(tag) 분석
"""
import sys
sys.path.insert(0, "/home/wavus/face_app/back_analysis/src")

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

from database.connection import DatabaseManager
from database.models import (
    ReferenceProfile,
    ReferenceTag,
    ReferenceAnalysisData,
)
import json
from pathlib import Path


# ==================== 정의 파일 로딩 ====================

def load_description_mapping():
    """JSON 정의 파일에서 ratio_name → description 매핑 로드"""
    mapping = {}

    # Atomic measurements
    atomic_dir = Path("/home/wavus/face_app/back_analysis/src/database/definitions/atomic_measurements")
    if atomic_dir.exists():
        for json_file in atomic_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'measurements' in data:
                        for m in data['measurements']:
                            mapping[m['measurement_name']] = m.get('description', m['measurement_name'])
            except Exception:
                pass

    # Derived ratios
    derived_dir = Path("/home/wavus/face_app/back_analysis/src/database/definitions/derived_ratios")
    if derived_dir.exists():
        for json_file in derived_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'ratios' in data:
                        for r in data['ratios']:
                            mapping[r['ratio_name']] = r.get('description', r['ratio_name'])
            except Exception:
                pass

    return mapping


# 전역 캐싱
_description_cache = None

def get_description(name: str) -> str:
    """ratio_name에서 description 반환 (캐싱)"""
    global _description_cache
    if _description_cache is None:
        _description_cache = load_description_mapping()
    return _description_cache.get(name, name)


def split_name_and_side(full_name: str) -> Tuple[str, str]:
    """이름에서 side 접미사(-left/-right/-center)를 분리하여 (base, side) 반환"""
    if not isinstance(full_name, str):
        return full_name, None
    for s in ("left", "right", "center"):
        suffix = f"-{s}"
        if full_name.endswith(suffix):
            return full_name[: -len(suffix)], s
    return full_name, None


# ==================== 데이터 로딩 ====================

def get_available_atomic_measurements() -> List[Dict]:
    """스냅샷(JSONB) 기반 사용 가능한 atomic 목록 반환"""
    db_manager = DatabaseManager()
    with db_manager.get_session() as session:
        snaps = session.query(ReferenceAnalysisData).filter(ReferenceAnalysisData.is_latest.is_(True)).all()
        keys = {}
        for s in snaps:
            atomic = (s.data_json or {}).get('atomic') or {}
            for name, meta in atomic.items():
                # name에서 side 추론 후, meta 정보와 병합
                base_name, inferred_side = split_name_and_side(name)
                side = meta.get('side') or inferred_side
                # description은 base name 기준으로 조회
                description = get_description(base_name)
                display = f"{description}{f' ({side})' if side else ''}"
                keys[name] = {
                    'display': display,
                    'name': name,  # 실제 키는 스냅샷 키 유지
                    'type': meta.get('measurement_type'),
                    'side': side,
                }
        return sorted(keys.values(), key=lambda x: x['display'])


def get_available_derived_measurements() -> List[Dict]:
    """스냅샷(JSONB) 기반 사용 가능한 derived 목록 반환"""
    db_manager = DatabaseManager()
    with db_manager.get_session() as session:
        snaps = session.query(ReferenceAnalysisData).filter(ReferenceAnalysisData.is_latest.is_(True)).all()
        keys = {}
        for s in snaps:
            derived = (s.data_json or {}).get('derived') or {}
            for name, meta in derived.items():
                # name에서 side 추론 후, meta 정보와 병합
                base_name, inferred_side = split_name_and_side(name)
                side = meta.get('side') or inferred_side
                cat = meta.get('category')
                # description은 base name 기준으로 조회
                description = get_description(base_name)
                side_str = f" ({side})" if side else ""
                cat_str = f" [{cat}]" if cat else ""
                display = f"{description}{side_str}{cat_str}"
                keys[name] = {
                    'display': display,
                    'name': name,  # 실제 키는 스냅샷 키 유지
                    'category': cat,
                    'side': side,
                }
        return sorted(keys.values(), key=lambda x: x['display'])


def get_available_tags() -> List[str]:
    """DB에서 사용 가능한 tag 목록 반환"""
    db_manager = DatabaseManager()
    with db_manager.get_session() as session:
        tags = session.query(ReferenceTag.tag_name).distinct().all()
        return sorted([t.tag_name for t in tags])


def prepare_statistical_dataset(independent_vars: List[Dict], target_tag: str) -> pd.DataFrame:
    """
    통계 분석용 데이터셋 준비

    Args:
        independent_vars: [
            {"type": "atomic", "name": "eye-width-left", "side": "left"},
            {"type": "derived", "name": "forehead-to-midface"},
            {"type": "tag", "name": "긴얼굴형"},
            ...
        ]
        target_tag: "긴얼굴형"

    Returns:
        DataFrame with columns:
        - profile_id
        - var_0, var_1, var_2, ... (독립변수들)
        - target (0 or 1) ← target_tag 보유 여부
    """
    db_manager = DatabaseManager()
    with db_manager.get_session() as session:
        # 모든 profile_id + 최신 스냅샷 로드
        profiles = session.query(ReferenceProfile.id).all()
        profile_ids = [p.id for p in profiles]
        snaps = session.query(ReferenceAnalysisData).filter(ReferenceAnalysisData.is_latest.is_(True)).all()
        snap_map = {s.profile_id: (s.data_json or {}) for s in snaps}

        # DataFrame 초기화
        df = pd.DataFrame({'profile_id': profile_ids})

        # 독립 변수 추가
        for idx, var in enumerate(independent_vars):
            col_name = f"var_{idx}"
            vtype = var.get('type')

            if vtype == 'atomic':
                name = var.get('name')
                side = var.get('side')
                # key 규칙: name에 side 미포함이면 suffix로 추가
                def build_key(n, s):
                    if not s:
                        return n
                    if n.endswith('-left') or n.endswith('-right') or n.endswith('-center'):
                        return n
                    return f"{n}-{s}"
                key = build_key(name, side)
                values = {}
                for pid in profile_ids:
                    data = snap_map.get(pid) or {}
                    atomic = data.get('atomic') or {}
                    meta = atomic.get(key)
                    if meta and meta.get('value') is not None:
                        try:
                            values[pid] = float(meta['value'])
                        except Exception:
                            pass
                df[col_name] = df['profile_id'].map(values)

            elif vtype == 'derived':
                name = var.get('name')
                side = var.get('side')
                def build_key(n, s):
                    if not s:
                        return n
                    if n.endswith('-left') or n.endswith('-right') or n.endswith('-center'):
                        return n
                    return f"{n}-{s}"
                key = build_key(name, side)
                values = {}
                for pid in profile_ids:
                    data = snap_map.get(pid) or {}
                    derived = data.get('derived') or {}
                    meta = derived.get(key)
                    if meta and meta.get('value') is not None:
                        try:
                            values[pid] = float(meta['value'])
                        except Exception:
                            pass
                df[col_name] = df['profile_id'].map(values)

            elif vtype == 'tag':
                # Tag 보유 여부 (0 or 1)
                tag_data = session.query(
                    ReferenceTag.profile_id
                ).filter(
                    ReferenceTag.tag_name == var['name']
                ).all()
                tag_profile_ids = {t.profile_id for t in tag_data}
                df[col_name] = df['profile_id'].apply(lambda pid: 1 if pid in tag_profile_ids else 0)

        # 종속 변수 (target) 추가
        target_data = session.query(ReferenceTag.profile_id).filter(ReferenceTag.tag_name == target_tag).all()
        target_profile_ids = {t.profile_id for t in target_data}
        df['target'] = df['profile_id'].apply(lambda pid: 1 if pid in target_profile_ids else 0)

        # NaN이 있는 행 제거
        df = df.dropna()
        return df


# ==================== 통계 기법 추천 ====================

def recommend_statistical_methods(num_vars: int) -> Dict[str, List[str]]:
    """
    독립변수 개수에 따라 적합한 통계 기법과 시각화 추천

    Returns:
        {
            "methods": ["Pearson", "Spearman", "t-test"],
            "visualizations": ["Scatter", "Box", "Bar"]
        }
    """
    if num_vars == 1:
        return {
            "methods": ["Pearson 상관계수", "Spearman 상관계수", "t-test"],
            "visualizations": ["Scatter Plot", "Box Plot", "Bar Chart"]
        }
    elif num_vars == 2:
        return {
            "methods": ["Multiple Regression", "ANOVA"],
            "visualizations": ["3D Scatter", "Heatmap", "Regression Plane"]
        }
    elif num_vars == 3:
        return {
            "methods": ["Random Forest", "SHAP"],
            "visualizations": ["Feature Importance", "PDP (Partial Dependence)", "Heatmap"]
        }
    else:  # 4개 이상
        return {
            "methods": ["PCA", "t-SNE", "Neural Network"],
            "visualizations": ["Embedding Plot", "Network Graph", "Parallel Coordinates"]
        }


# ==================== 1:1 분석 ====================

def analyze_1to1_pearson(df: pd.DataFrame, var_name: str) -> Dict:
    """Pearson 상관계수 분석"""
    corr, p_value = stats.pearsonr(df['var_0'], df['target'])

    return {
        "correlation": corr,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def analyze_1to1_spearman(df: pd.DataFrame, var_name: str) -> Dict:
    """Spearman 상관계수 분석"""
    corr, p_value = stats.spearmanr(df['var_0'], df['target'])

    return {
        "correlation": corr,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def analyze_1to1_ttest(df: pd.DataFrame, var_name: str) -> Dict:
    """t-test 분석"""
    group_0 = df[df['target'] == 0]['var_0']
    group_1 = df[df['target'] == 1]['var_0']

    t_stat, p_value = stats.ttest_ind(group_0, group_1)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "mean_group_0": group_0.mean(),
        "mean_group_1": group_1.mean()
    }


def visualize_1to1(df: pd.DataFrame, var_name: str, viz_type: str):
    """1:1 시각화"""
    if viz_type == "Scatter Plot":
        fig = px.scatter(df, x='var_0', y='target',
                        title=f"Scatter Plot: {var_name} vs Target",
                        labels={'var_0': var_name, 'target': 'Target (0/1)'},
                        trendline="ols")
        return fig

    elif viz_type == "Box Plot":
        df['target_label'] = df['target'].map({0: 'Without Tag', 1: 'With Tag'})
        fig = px.box(df, x='target_label', y='var_0',
                    title=f"Box Plot: {var_name} by Target",
                    labels={'var_0': var_name, 'target_label': 'Target'})
        return fig

    elif viz_type == "Bar Chart":
        # target별 평균값
        avg_data = df.groupby('target')['var_0'].mean().reset_index()
        avg_data['target_label'] = avg_data['target'].map({0: 'Without Tag', 1: 'With Tag'})
        fig = px.bar(avg_data, x='target_label', y='var_0',
                    title=f"Mean {var_name} by Target",
                    labels={'var_0': f'Mean {var_name}', 'target_label': 'Target'})
        return fig


# ==================== 2:1 분석 ====================

def analyze_2to1_regression(df: pd.DataFrame, var_names: List[str]) -> Dict:
    """Multiple Regression 분석"""
    X = df[['var_0', 'var_1']]
    y = df['target']

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2_score = model.score(X, y)

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "r2_score": r2_score,
        "var_names": var_names
    }


def analyze_2to1_anova(df: pd.DataFrame, var_names: List[str]) -> Dict:
    """2-way ANOVA 분석"""
    # 간단한 ANOVA 구현 (statsmodels 사용 시 더 정교함)
    group_0 = df[df['target'] == 0][['var_0', 'var_1']]
    group_1 = df[df['target'] == 1][['var_0', 'var_1']]

    f_stat_0, p_value_0 = stats.f_oneway(group_0['var_0'], group_1['var_0'])
    f_stat_1, p_value_1 = stats.f_oneway(group_0['var_1'], group_1['var_1'])

    return {
        "var_0_f_stat": f_stat_0,
        "var_0_p_value": p_value_0,
        "var_1_f_stat": f_stat_1,
        "var_1_p_value": p_value_1
    }


def visualize_2to1(df: pd.DataFrame, var_names: List[str], viz_type: str):
    """2:1 시각화"""
    if viz_type == "3D Scatter":
        df['target_label'] = df['target'].map({0: 'Without Tag', 1: 'With Tag'})
        fig = px.scatter_3d(df, x='var_0', y='var_1', z='target',
                           color='target_label',
                           title=f"3D Scatter: {var_names[0]} vs {var_names[1]} vs Target",
                           labels={'var_0': var_names[0], 'var_1': var_names[1], 'target': 'Target'})
        return fig

    elif viz_type == "Heatmap":
        # 상관관계 매트릭스
        corr_matrix = df[['var_0', 'var_1', 'target']].corr()
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       title="Correlation Heatmap",
                       labels={'color': 'Correlation'},
                       x=['var_0', 'var_1', 'target'],
                       y=['var_0', 'var_1', 'target'])
        return fig


# ==================== 3:1 분석 ====================

def analyze_3to1_random_forest(df: pd.DataFrame, var_names: List[str]) -> Dict:
    """Random Forest 분석"""
    X = df[['var_0', 'var_1', 'var_2']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importances = model.feature_importances_

    return {
        "accuracy": accuracy,
        "feature_importances": feature_importances.tolist(),
        "var_names": var_names,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


def visualize_3to1(df: pd.DataFrame, var_names: List[str], viz_type: str, analysis_result: Dict = None):
    """3:1 시각화"""
    if viz_type == "Feature Importance":
        if analysis_result and 'feature_importances' in analysis_result:
            importance_df = pd.DataFrame({
                'Feature': var_names,
                'Importance': analysis_result['feature_importances']
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Importance',
                        title="Feature Importance (Random Forest)",
                        labels={'Importance': 'Importance Score'})
            return fig

    elif viz_type == "Heatmap":
        corr_matrix = df[['var_0', 'var_1', 'var_2', 'target']].corr()
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       title="Correlation Heatmap",
                       labels={'color': 'Correlation'})
        return fig


# ==================== 4:1+ 분석 ====================

def analyze_4plus_pca(df: pd.DataFrame, var_names: List[str]) -> Dict:
    """PCA 차원 축소 분석"""
    var_cols = [f'var_{i}' for i in range(len(var_names))]
    X = df[var_cols]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "X_pca": X_pca.tolist(),
        "var_names": var_names
    }


def analyze_4plus_tsne(df: pd.DataFrame, var_names: List[str]) -> Dict:
    """t-SNE 차원 축소 분석"""
    var_cols = [f'var_{i}' for i in range(len(var_names))]
    X = df[var_cols]

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    return {
        "X_tsne": X_tsne.tolist(),
        "var_names": var_names
    }


def visualize_4plus(df: pd.DataFrame, var_names: List[str], viz_type: str, analysis_result: Dict = None):
    """4:1+ 시각화"""
    if viz_type == "Embedding Plot":
        if analysis_result and 'X_pca' in analysis_result:
            # PCA 결과
            pca_df = pd.DataFrame(analysis_result['X_pca'], columns=['PC1', 'PC2'])
            pca_df['target'] = df['target'].values
            pca_df['target_label'] = pca_df['target'].map({0: 'Without Tag', 1: 'With Tag'})

            fig = px.scatter(pca_df, x='PC1', y='PC2', color='target_label',
                           title="PCA Embedding Plot",
                           labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
            return fig

        elif analysis_result and 'X_tsne' in analysis_result:
            # t-SNE 결과
            tsne_df = pd.DataFrame(analysis_result['X_tsne'], columns=['Dim1', 'Dim2'])
            tsne_df['target'] = df['target'].values
            tsne_df['target_label'] = tsne_df['target'].map({0: 'Without Tag', 1: 'With Tag'})

            fig = px.scatter(tsne_df, x='Dim1', y='Dim2', color='target_label',
                           title="t-SNE Embedding Plot",
                           labels={'Dim1': 't-SNE Dimension 1', 'Dim2': 't-SNE Dimension 2'})
            return fig

    elif viz_type == "Parallel Coordinates":
        var_cols = [f'var_{i}' for i in range(len(var_names))]
        plot_df = df[var_cols + ['target']].copy()
        plot_df['target_label'] = plot_df['target'].map({0: 'Without Tag', 1: 'With Tag'})

        fig = px.parallel_coordinates(plot_df,
                                     color='target',
                                     dimensions=var_cols,
                                     title="Parallel Coordinates Plot",
                                     labels={col: var_names[i] for i, col in enumerate(var_cols)})
        return fig
