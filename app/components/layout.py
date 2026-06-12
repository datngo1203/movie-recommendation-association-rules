import streamlit as st

from config import APP_ICON, APP_TITLE
from recommendation_service import RuleSource


def render_app_header() -> None:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Ứng dụng minh họa Chương 3: phát triển giao diện, trình bày kết quả "
        "và chuẩn bị điểm tích hợp cho Apriori, FP-Growth."
    )


def render_sidebar(rule_sources: dict[str, RuleSource]) -> dict[str, object]:
    st.sidebar.header("Cấu hình gợi ý")

    algorithm = st.sidebar.radio(
        "Thuật toán",
        options=list(rule_sources.keys()),
        horizontal=True,
    )
    min_confidence = st.sidebar.slider(
        "Ngưỡng confidence tối thiểu",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    max_results = st.sidebar.number_input(
        "Số lượng phim gợi ý",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.sidebar.divider()
    st.sidebar.caption("Trạng thái tích hợp")
    for source in rule_sources.values():
        status = "Sẵn sàng" if source.is_available else "Chưa có file"
        st.sidebar.write(f"{source.algorithm}: {status}")

    return {
        "algorithm": algorithm,
        "min_confidence": float(min_confidence),
        "max_results": int(max_results),
    }
