import argparse
import json
import time

from src.pipeline import run_pipeline, get_query_validation


def get_recommendations(query, validate=True):
    """
    Get BIS standard recommendations for a query.
    
    Args:
        query (str): Product description
        validate (bool): Validate if query is building-material related
        
    Returns:
        list: Standard identifiers
    """
    return run_pipeline(query, top_k=5, validate=validate)


def main():
    parser = argparse.ArgumentParser(description="Run BIS standard recommendations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Enable LLM validation of building-material queries (requires Ollama)"
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    for item in data:
        start_time = time.perf_counter()
        
        # Get recommendations with optional validation
        standards = get_recommendations(item["query"], validate=args.validate)
        
        latency = time.perf_counter() - start_time

        result = {
            "id": item["id"],
            "retrieved_standards": standards[:5],
            "latency_seconds": latency,
        }
        
        # Include optional fields if present in input
        if "query" in item:
            result["query"] = item["query"]
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]
        
        # Include validation info if validation was enabled
        if args.validate and len(standards) == 0:
            validation = get_query_validation(item["query"])
            result["validation_message"] = validation["message"]
        
        results.append(result)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
