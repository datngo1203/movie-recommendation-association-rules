import pandas as pd
import streamlit as st

from recommendation_service import RuleSource


def render_dataset_overview(movies_df: pd.DataFrame) -> None:
    st.subheader("Dữ liệu phim")

    total_movies = len(movies_df)
    genres = _count_unique_genres(movies_df)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Số phim", f"{total_movies:,}")
    metric_columns[1].metric("Số thể loại", f"{genres:,}")
    metric_columns[2].metric("Nguồn dữ liệu", "MovieLens")

    if movies_df.empty:
        st.warning("Chưa tìm thấy file data/raw/movies.csv.")
        return

    st.dataframe(movies_df.head(20), use_container_width=True, hide_index=True)


def render_model_readiness(rule_sources: dict[str, RuleSource]) -> None:
    st.subheader("Trạng thái kết quả mô hình")

    rows = [
        {
            "Thuật toán": source.algorithm,
            "File mong đợi": str(source.path),
            "Trạng thái": "Đã có dữ liệu" if source.is_available else "Chưa có dữ liệu",
            "Số luật": len(source.rules),
        }
        for source in rule_sources.values()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _count_unique_genres(movies_df: pd.DataFrame) -> int:
    if "genres" not in movies_df.columns or movies_df.empty:
        return 0

    unique_genres = set()
    for value in movies_df["genres"].dropna():
        unique_genres.update(genre for genre in str(value).split("|") if genre)
    return len(unique_genres)
