from src.retriever import get_retriever
from src.llm_classifier import validate_query


def run_pipeline(query, top_k=5, include_rationale=False, validate=True):
    """
    Return the top BIS standard identifiers for a product description.
    
    Args:
        query (str): Product description
        top_k (int): Number of standards to return
        include_rationale (bool): Include title as rationale
        validate (bool): Use LLM to validate if query is building-material related
        
    Returns:
        list: Standard identifiers or empty list if query is invalid
    """
    # Validate query if requested
    if validate:
        validation = validate_query(query)
        if not validation["is_valid"]:
            # Return empty list for invalid queries
            return []
    
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


def get_query_validation(query):
    """Get validation result without running full pipeline."""
    return validate_query(query)
