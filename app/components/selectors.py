import pandas as pd
import streamlit as st


def render_movie_selector(movies_df: pd.DataFrame) -> list[str]:
    st.subheader("Phim nguoi dung da xem")

    if movies_df.empty or "title" not in movies_df.columns:
        st.warning("Danh sach phim chua san sang.")
        return []

    movie_titles = movies_df["title"].dropna().sort_values().unique().tolist()
    selected_titles = st.multiselect(
        "Chon mot hoac nhieu phim trong dataset",
        options=movie_titles,
        placeholder="Nhap ten phim de tim kiem",
    )
    custom_titles = st.text_area(
        "Nhap phim khac neu khong co trong danh sach",
        placeholder="Moi dong mot ten phim, vi du: Avatar 3 (2025)",
        height=80,
    )
    typed_titles = [
        title.strip()
        for title in custom_titles.splitlines()
        if title.strip()
    ]

    return list(dict.fromkeys([*selected_titles, *typed_titles]))


def render_genre_selector(movies_df: pd.DataFrame) -> list[str]:
    st.subheader("Thể loại phim yêu thích")

    if movies_df.empty or "genres" not in movies_df.columns:
        st.warning("Danh sách thể loại chưa sẵn sàng.")
        return []

    # Extract all unique genres from the dataframe
    unique_genres = set()
    for value in movies_df["genres"].dropna():
        unique_genres.update(genre.strip() for genre in str(value).split("|") if genre.strip() and genre.strip() != "(no genres listed)")
    
    sorted_genres = sorted(list(unique_genres))

    selected_genres = st.multiselect(
        "Chọn một hoặc nhiều thể loại bạn quan tâm",
        options=sorted_genres,
        placeholder="Chọn thể loại phim",
    )

    return selected_genres

