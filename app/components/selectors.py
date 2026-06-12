import pandas as pd
import streamlit as st


def render_movie_selector(movies_df: pd.DataFrame) -> list[str]:
    st.subheader("Phim người dùng đã xem")

    if movies_df.empty or "title" not in movies_df.columns:
        st.warning("Danh sách phim chưa sẵn sàng.")
        return []

    movie_titles = movies_df["title"].dropna().sort_values().unique().tolist()
    return st.multiselect(
        "Chọn một hoặc nhiều phim",
        options=movie_titles,
        placeholder="Nhập tên phim để tìm kiếm",
    )
