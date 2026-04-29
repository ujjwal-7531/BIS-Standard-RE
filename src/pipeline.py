from src.retriever import get_retriever


def run_pipeline(query, top_k=5, include_rationale=False):
    """Return the top BIS standard identifiers for a product description."""
    hits = get_retriever().retrieve(query, top_k=top_k)
    if include_rationale:
        return [
            {
                "standard": hit["standard"],
                "rationale": f"Matched against: {hit['title']}",
            }
            for hit in hits
        ]
    return [hit["standard"] for hit in hits]
