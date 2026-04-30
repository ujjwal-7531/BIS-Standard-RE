"""
LLM-based query classifier to determine if a query is building-material related.
Uses Ollama with phi:2.7b model for fast, lightweight classification.
"""

import ollama
from functools import lru_cache


BUILDING_MATERIAL_KEYWORDS = {
    "cement",
    "concrete",
    "steel",
    "aggregate",
    "brick",
    "block",
    "tile",
    "mortar",
    "sand",
    "gravel",
    "asbestos",
    "gypsum",
    "lime",
    "pozzolana",
    "fly ash",
    "slag",
    "pipe",
    "precast",
    "masonry",
    "timber",
    "roofing",
    "cladding",
    "plaster",
    "coating",
    "paint",
    "adhesive",
    "sealant",
    "insulation",
    "glass",
    "window",
    "door",
    "frame",
    "fastener",
    "bolt",
    "weld",
    "rebar",
    "reinforcement",
    "formwork",
    "scaffold",
    "safety",
    "ppe",
    "equipment",
    "machinery",
    "building",
    "construction",
    "structure",
    "foundation",
    "beam",
    "column",
    "slab",
    "wall",
    "floor",
    "roof",
    "stair",
}


def has_material_keywords(query):
    """Quick keyword-based check before LLM call."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in BUILDING_MATERIAL_KEYWORDS)


@lru_cache(maxsize=128)
def classify_query_with_llm(query):
    """
    Use Ollama (phi:2.7b) to classify if query is building-material related.
    
    Returns:
        tuple: (is_relevant: bool, confidence: str)
    """
    try:
        # Quick keyword check first - if match, skip LLM call
        if has_material_keywords(query):
            return True, "keyword_match"
        
        # Call Ollama classifier
        prompt = f"""You are a building materials classification expert. 
Determine if this query is about building materials (cement, steel, concrete, bricks, aggregates, etc.) or related to construction/building standards.

Query: "{query}"

Answer with only "YES" or "NO" (nothing else):"""
        
        response = ollama.generate(
            model="phi",
            prompt=prompt,
            stream=False,
        )
        
        answer = response["response"].strip().upper()
        is_relevant = "YES" in answer
        
        return is_relevant, "llm_classified"
        
    except Exception as e:
        # If Ollama fails, fall back to keyword matching
        print(f"Warning: LLM classification failed ({e}), using keyword fallback")
        return has_material_keywords(query), "fallback"


def validate_query(query):
    """
    Validate if query is building-material related.
    
    Args:
        query (str): User's product description query
        
    Returns:
        dict: {
            "is_valid": bool,
            "confidence": str,
            "message": str (if not valid)
        }
    """
    if not query or len(query.strip()) < 3:
        return {
            "is_valid": False,
            "confidence": "too_short",
            "message": "Query too short. Please provide a more detailed product description."
        }
    
    is_relevant, method = classify_query_with_llm(query)
    
    if is_relevant:
        return {
            "is_valid": True,
            "confidence": method,
            "message": "Query is building-material related"
        }
    else:
        return {
            "is_valid": False,
            "confidence": method,
            "message": "Query does not appear to be about building materials. Please describe a building material product."
        }
