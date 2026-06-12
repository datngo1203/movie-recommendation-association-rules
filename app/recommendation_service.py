from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from config import RULE_OUTPUTS


@dataclass(frozen=True)
class RuleSource:
    algorithm: str
    path: Path
    is_available: bool
    rules: pd.DataFrame


@dataclass(frozen=True)
class RecommendationRequest:
    selected_movie_titles: list[str]
    algorithm: str
    min_confidence: float
    max_results: int


@dataclass(frozen=True)
class RecommendationResult:
    request: RecommendationRequest
    recommendations: pd.DataFrame
    matched_rules: pd.DataFrame
    message: str


@st.cache_data(show_spinner=False)
def _read_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_rule_sources() -> dict[str, RuleSource]:
    sources = {}
    for algorithm, path in RULE_OUTPUTS.items():
        rules = _read_rules(path)
        sources[algorithm] = RuleSource(
            algorithm=algorithm,
            path=path,
            is_available=path.exists() and not rules.empty,
            rules=rules,
        )
    return sources


def build_recommendations(
    request: RecommendationRequest,
    rule_sources: dict[str, RuleSource],
) -> RecommendationResult:
    if not request.selected_movie_titles:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=["movie", "score", "source_rule"]),
            matched_rules=pd.DataFrame(),
            message="Vui lòng chọn ít nhất một phim đã xem.",
        )

    selected_source = rule_sources.get(request.algorithm)
    if selected_source is None or not selected_source.is_available:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=["movie", "score", "source_rule"]),
            matched_rules=pd.DataFrame(),
            message=(
                "Chưa có dữ liệu luật kết hợp cho thuật toán đã chọn. "
                "Hãy xuất kết quả mô hình từ Chương 2 trước khi tích hợp logic gợi ý."
            ),
        )

    rules = selected_source.rules.copy()

    recommendations = []

    for movie in request.selected_movie_titles:
        matched = rules[
            rules["antecedents"].astype(str).str.strip()
            == movie.strip()
        ]

        for _, row in matched.iterrows():
            recommendations.append(
                {
                    "movie": row["consequents"],
                    "score": row["confidence"],
                    "source_rule": row["antecedents"],
                }
            )

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(),
            matched_rules=pd.DataFrame(),
            message="Không tìm thấy gợi ý phù hợp.",
        )

    recommendations_df = (
        recommendations_df
        .sort_values(by="score", ascending=False)
        .drop_duplicates(subset=["movie"])
        .head(request.max_results)
    )

    return RecommendationResult(
        request=request,
        recommendations=recommendations_df,
        matched_rules=rules,
        message=f"Tìm thấy {len(recommendations_df)} phim gợi ý.",
    )
        
