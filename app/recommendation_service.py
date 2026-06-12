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
    selected_genres: list[str]
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
        actual_path = path
        if not actual_path.exists():
            if algorithm == "FP-Growth" and (path.parent / "fpgrowth.csv").exists():
                actual_path = path.parent / "fpgrowth.csv"
            elif algorithm == "Apriori" and (path.parent / "apriori.csv").exists():
                actual_path = path.parent / "apriori.csv"

        rules = _read_rules(actual_path)
        sources[algorithm] = RuleSource(
            algorithm=algorithm,
            path=actual_path,
            is_available=actual_path.exists() and not rules.empty,
            rules=rules,
        )
    return sources


def build_recommendations(
    request: RecommendationRequest,
    rule_sources: dict[str, RuleSource],
) -> RecommendationResult:
    empty_columns = ["movie", "genres", "score", "source_rule", "algorithm", "support", "lift"]

    if not request.selected_movie_titles and not request.selected_genres:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=empty_columns),
            matched_rules=pd.DataFrame(),
            message="Vui lòng chọn phim đã xem hoặc thể loại phim yêu thích.",
        )

    selected_source = rule_sources.get(request.algorithm)
    if selected_source is None or not selected_source.is_available:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=empty_columns),
            matched_rules=pd.DataFrame(),
            message=(
                "Chưa có dữ liệu luật kết hợp cho thuật toán đã chọn. "
                "Hãy chạy mô hình trước khi thực hiện gợi ý."
            ),
        )

    rules = selected_source.rules.copy()
    rules = _normalize_rule_columns(rules)
    required_columns = {"watched_movie", "recommended_movie", "confidence"}
    if not required_columns.issubset(rules.columns):
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=empty_columns),
            matched_rules=pd.DataFrame(),
            message="File luật thiếu cột bắt buộc: watched_movie, recommended_movie, confidence.",
        )

    rules = rules[rules["confidence"] >= request.min_confidence]

    from data_loader import load_movies
    movies_df = load_movies()

    # Helper function for Jaccard Similarity
    def get_jaccard_similarity(genres_a_str, genres_b_str):
        set_a = set(g.strip() for g in str(genres_a_str).split("|") if g.strip() and g.strip() != "(no genres listed)")
        set_b = set(g.strip() for g in str(genres_b_str).split("|") if g.strip() and g.strip() != "(no genres listed)")
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    recommendations = []
    matched_rules_list = []

    # Map movie titles to their genres for quick lookup
    movie_to_genres = dict(zip(movies_df["title"], movies_df["genres"]))

    # CASE 1: The user selected at least one movie
    if request.selected_movie_titles:
        # Find rules where watched_movie is in request.selected_movie_titles
        matched_rules = rules[rules["watched_movie"].astype(str).str.strip().isin(request.selected_movie_titles)]
        
        if not matched_rules.empty:
            matched_rules_list.append(matched_rules)
            
            for _, rule_row in matched_rules.iterrows():
                rec_movie = rule_row["recommended_movie"]
                watched_movie = rule_row["watched_movie"]
                confidence = rule_row["confidence"]
                
                # Exclude already watched movies from recommendations
                if rec_movie in request.selected_movie_titles:
                    continue
                
                genres_watched = movie_to_genres.get(watched_movie, "")
                genres_rec = movie_to_genres.get(rec_movie, "")
                
                # Filter by selected genres from the box (if any are selected)
                if request.selected_genres:
                    rec_genres_set = set(g.strip() for g in str(genres_rec).split("|") if g.strip())
                    if not rec_genres_set.intersection(request.selected_genres):
                        continue
                
                # Compute Jaccard genre similarity between the watched movie and recommended movie
                jaccard = get_jaccard_similarity(genres_watched, genres_rec)
                
                # Compute combined score: confidence * (0.2 + 0.8 * jaccard)
                score = confidence * (0.2 + 0.8 * jaccard)
                
                # Find the movie's rating statistics
                movie_match = movies_df[movies_df["title"] == rec_movie]
                rating_count = 0
                rating_mean = 0.0
                if not movie_match.empty:
                    rating_count = movie_match.iloc[0].get("rating_count", 0)
                    rating_mean = movie_match.iloc[0].get("rating_mean", 0.0)
                
                recommendations.append({
                    "movie": rec_movie,
                    "genres": genres_rec,
                    "score": round(score, 4),
                    "source_rule": f"{watched_movie} -> {rec_movie}",
                    "algorithm": request.algorithm,
                    "support": rule_row["support"],
                    "lift": rule_row["lift"],
                    "rating_count": rating_count,
                    "rating_mean": rating_mean,
                })

    # CASE 2: The user only selected genres (no movies selected)
    elif request.selected_genres:
        # Recommend top popular movies belonging to these genres
        for _, movie_row in movies_df.iterrows():
            movie_title = movie_row["title"]
            movie_genres = movie_row["genres"]
            rec_genres_set = set(g.strip() for g in str(movie_genres).split("|") if g.strip())
            
            if rec_genres_set.intersection(request.selected_genres):
                recommendations.append({
                    "movie": movie_title,
                    "genres": movie_genres,
                    "score": 1.0,  # Base score for category suggestion
                    "source_rule": "Gợi ý trực tiếp từ thể loại",
                    "algorithm": request.algorithm,
                    "support": None,
                    "lift": None,
                    "rating_count": movie_row.get("rating_count", 0),
                    "rating_mean": movie_row.get("rating_mean", 0.0),
                })

    recommendations_df = pd.DataFrame(recommendations, columns=empty_columns + ["rating_count", "rating_mean"])
    matched_rules_df = pd.concat(matched_rules_list, ignore_index=True) if matched_rules_list else pd.DataFrame(columns=rules.columns)

    if recommendations_df.empty:
        return RecommendationResult(
            request=request,
            recommendations=pd.DataFrame(columns=empty_columns),
            matched_rules=matched_rules_df,
            message="Không tìm thấy phim gợi ý nào phù hợp.",
        )

    # Sort and filter duplicate recommendations (keep highest score)
    recommendations_df = (
        recommendations_df
        .sort_values(by=["score", "rating_count", "rating_mean"], ascending=[False, False, False])
        .drop_duplicates(subset=["movie"])
        .head(request.max_results)
    )

    recommendations_df = recommendations_df[empty_columns]

    message = f"Tìm thấy {len(recommendations_df)} phim gợi ý phù hợp."

    return RecommendationResult(
        request=request,
        recommendations=recommendations_df,
        matched_rules=matched_rules_df,
        message=message,
    )


def _normalize_rule_columns(rules: pd.DataFrame) -> pd.DataFrame:
    return rules.rename(
        columns={
            "antecedents": "watched_movie",
            "consequents": "recommended_movie",
        }
    )


