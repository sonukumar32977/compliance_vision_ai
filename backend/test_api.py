"""
Quick local test for Compliance Vision retrieval API.
Run the server first:  uvicorn api_server:app --host 127.0.0.1 --port 8001
Then:                python test_api.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"


def get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    print("=" * 60)
    print("Compliance Vision API test")
    print("=" * 60)

    try:
        health = get("/health")
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Cannot reach {BASE}")
        print("Start the server: uvicorn api_server:app --host 127.0.0.1 --port 8001")
        print(e)
        sys.exit(1)

    print(f"\n[1] Health: {health['status']}")
    print(f"    Vectors indexed: {health['indexed_vectors']}")
    print(f"    Embedding model: {health['embedding_model']}")

    print("\n[2] Single query test (POST /retrieve)")
    query = "worker without helmet in restricted area"
    result = post("/retrieve", {"query": query, "top_k": 3})
    m = result["metrics"]
    print(f"    Query: {query}")
    print(f"    Chunks found: {result['chunk_count']}")
    print(f"    Speed: {m['total_ms']} ms (embed {m['embed_ms']} ms + search {m['search_ms']} ms)")
    print(f"    Scores: avg={m['avg_score']} max={m['max_score']}")
    print(f"    Efficiency: {m['efficiency']}")
    if result["chunks"]:
        top = result["chunks"][0]
        print(f"    Top hit: {top['source_file']} (score {top['score']})")
        print(f"    Preview: {top['text'][:120]}...")

    print("\n[3] Benchmark (GET /benchmark — 5 sample queries)")
    bench = get("/benchmark")
    print(f"    Overall: {bench['overall_efficiency']}")
    print(f"    Avg latency: {bench['average_latency_ms']} ms")
    print(f"    Avg max score: {bench['average_max_score']}")
    for run in bench["runs"]:
        print(f"    - {run['query'][:45]:<45} | chunks={run['chunk_count']} max={run['max_score']} {run['total_ms']}ms")

    print("\n[OK] API is working. Open Swagger for interactive tests:")
    print(f"    {BASE}/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
