import time

import streamlit as st

from src.retriever import get_retriever


EXAMPLE_QUERIES = [
    "We manufacture 33 Grade Ordinary Portland Cement and need the applicable BIS standard.",
    "Looking for the standard for precast concrete pipes for water mains and sewerage.",
    "We produce hollow and solid lightweight concrete masonry blocks.",
    "Which standard covers corrugated and semi-corrugated asbestos cement sheets for roofing?",
    "We make calcined clay based Portland pozzolana cement.",
]


st.set_page_config(
    page_title="BIS Standards Recommendation Engine",
    page_icon="BIS",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
    }
    .standard-card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        background: #ffffff;
    }
    .standard-id {
        font-size: 1.05rem;
        font-weight: 700;
        color: #102a43;
        margin-bottom: 0.25rem;
    }
    .standard-title {
        color: #334e68;
        font-size: 0.95rem;
        line-height: 1.35;
    }
    .score {
        color: #627d98;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_result(rank, hit):
    title = hit.get("title") or "Relevant BIS building-material standard"
    score = hit.get("score", 0.0)
    st.markdown(
        f"""
        <div class="standard-card">
            <div class="standard-id">{rank}. {hit["standard"]}</div>
            <div class="standard-title">{title}</div>
            <div class="score">Retrieval score: {score:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("BIS Standards Recommendation Engine")

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    query = st.text_area(
        "Product description",
        height=150,
        placeholder="Describe the material, product, grade, application, or compliance need...",
    )
    submitted = st.button("Recommend Standards", type="primary", use_container_width=True)

with right:
    selected_example = st.selectbox("Demo query", EXAMPLE_QUERIES)
    use_example = st.button("Use Demo Query", use_container_width=True)
    st.metric("Indexed BIS standards", len(get_retriever().documents))

if use_example:
    query = selected_example
    submitted = True

if submitted:
    if not query.strip():
        st.warning("Enter a product description first.")
    else:
        started = time.perf_counter()
        hits = get_retriever().retrieve(query, top_k=5)
        latency = time.perf_counter() - started

        st.subheader("Top Recommendations")
        st.caption(f"Latency: {latency:.3f} seconds")

        for rank, hit in enumerate(hits, start=1):
            render_result(rank, hit)
else:
    st.subheader("Top Recommendations")
    st.info("Enter a product description or choose a demo query.")
