import streamlit as st

from recommendation_service import RecommendationResult


def render_empty_recommendations() -> None:
    st.info("Chọn các phim đã xem và bấm Tạo gợi ý để xem kết quả.")


def render_recommendation_results(result: RecommendationResult) -> None:
    if result.recommendations.empty:
        st.warning(result.message)
        return

    st.success(result.message)
    st.dataframe(result.recommendations, use_container_width=True, hide_index=True)

    with st.expander("Luật kết hợp phù hợp"):
        st.dataframe(result.matched_rules, use_container_width=True, hide_index=True)
