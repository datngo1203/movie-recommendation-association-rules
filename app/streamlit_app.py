import streamlit as st

from components.layout import render_app_header, render_sidebar
from components.metrics import render_dataset_overview, render_model_readiness
from components.results import render_empty_recommendations, render_recommendation_results
from components.selectors import render_movie_selector, render_genre_selector
from config import APP_ICON, APP_TITLE
from data_loader import load_movies
from recommendation_service import RecommendationRequest, build_recommendations, load_rule_sources


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    render_app_header()

    movies_df = load_movies()
    rule_sources = load_rule_sources()
    settings = render_sidebar(rule_sources)

    overview_tab, recommender_tab, rules_tab = st.tabs(
        ["Tổng quan dữ liệu", "Gợi ý phim", "Kết quả luật kết hợp"]
    )

    with overview_tab:
        render_dataset_overview(movies_df)
        render_model_readiness(rule_sources)

    with recommender_tab:
        col1, col2 = st.columns(2)
        with col1:
            selected_movies = render_movie_selector(movies_df)
        with col2:
            selected_genres = render_genre_selector(movies_df)

        if st.button("Tạo gợi ý", type="primary", use_container_width=True):
            request = RecommendationRequest(
                selected_movie_titles=selected_movies,
                selected_genres=selected_genres,
                algorithm=settings["algorithm"],
                min_confidence=settings["min_confidence"],
                max_results=settings["max_results"],
            )
            result = build_recommendations(request, rule_sources)
            render_recommendation_results(result)
        else:
            render_empty_recommendations()

    with rules_tab:
        st.subheader("Đầu ra mô hình")

        st.info(
            "Khu vực này hiển thị tập luật kết hợp sinh ra từ Apriori và FP-Growth."
        )

        render_model_readiness(rule_sources)

        if rule_sources["Apriori"].is_available:
            st.markdown("### Luật Apriori")
            st.dataframe(
                rule_sources["Apriori"].rules,
                use_container_width=True,
                hide_index=True
            )

        if rule_sources["FP-Growth"].is_available:
            st.markdown("### Luật FP-Growth")
            st.dataframe(
                rule_sources["FP-Growth"].rules,
                use_container_width=True,
                hide_index=True
            )


if __name__ == "__main__":
    main()
