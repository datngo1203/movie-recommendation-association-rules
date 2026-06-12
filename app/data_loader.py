import pandas as pd
import streamlit as st

from config import MOVIES_FILE


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    if not MOVIES_FILE.exists():
        return pd.DataFrame(columns=["movieId", "title", "genres", "rating_mean", "rating_count"])

    movies = pd.read_csv(MOVIES_FILE)
    required_columns = ["movieId", "title", "genres", "rating_mean", "rating_count"]
    return movies[[column for column in required_columns if column in movies.columns]]
