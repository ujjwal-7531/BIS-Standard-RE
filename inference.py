import argparse
import json
import time

from src.pipeline import run_pipeline


def get_recommendations(query):
    return run_pipeline(query, top_k=5)


def main():
    parser = argparse.ArgumentParser(description="Run BIS standard recommendations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    for item in data:
        start_time = time.perf_counter()
        standards = get_recommendations(item["query"])
        latency = time.perf_counter() - start_time

        result = {
            "id": item["id"],
            "retrieved_standards": standards[:5],
            "latency_seconds": latency,
        }
        if "query" in item:
            result["query"] = item["query"]
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]
        results.append(result)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
