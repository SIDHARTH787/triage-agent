"""
Benchmark: sequential vs speculative SLM latency.
Feeds canned (partial, final) transcript pairs directly into both code
paths for reproducible, isolated measurement of the SLM-stage speedup.
"""

import time
import statistics
import yaml

from src.slm.slm_engine import SLMEngine
from src.speculative.speculator import Speculator
from src.speculative.divergence_detector import compute_divergence


TEST_CASES = [
    ("I am having a mild cold", "I am having a mild cold", []),
    ("I have a bad headache", "I have a bad headache", []),
    ("I am having a mild cord", "I am having a mild cold", []),
    ("makes my symptoms", "makes my symptoms worse", []),
    ("Using a humidifier", "Using a humidifier works", []),
    ("I have had a b-", "I have had a bad cough for 3 days", []),
    ("I have chest", "I have chest pain and difficulty breathing", []),
    ("I am having a blog", "I am having a blocked nose", []),
    ("Using a humidifier", "No", []),
    ("Hello how are you", "I am having a mild cough", []),
]


def benchmark_sequential(slm, final_text, history):
    t0 = time.time()
    slm.generate(final_text, history)
    return (time.time() - t0) * 1000


def benchmark_speculative(speculator, partial_text, final_text, history,
                           divergence_threshold, simulated_speaking_delay_s=1.5):
    speculator.start_speculation(partial_text, history)
    time.sleep(simulated_speaking_delay_s)

    t0 = time.time()
    divergence = compute_divergence(partial_text, final_text, threshold=divergence_threshold)

    if divergence.diverged:
        speculator.cancel()
        speculator.engine.generate(final_text, history)
        committed = False
    else:
        result = speculator.resolve(final_text)
        committed = result is not None
        if not committed:
            speculator.engine.generate(final_text, history)

    effective_latency_ms = (time.time() - t0) * 1000
    return {
        "committed": committed,
        "similarity": divergence.similarity,
        "effective_latency_ms": effective_latency_ms,
    }


def run_benchmark(config_path="config.yaml", divergence_threshold=0.7,
                   simulated_speaking_delay_s=1.5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    slm = SLMEngine(cfg["slm"])
    speculator = Speculator(cfg["slm"])

    sequential_latencies = []
    speculative_results = []

    print(f"Running {len(TEST_CASES)} test cases...")
    print(f"Divergence threshold: {divergence_threshold} | Simulated speaking delay: {simulated_speaking_delay_s}s\n")

    for i, (partial, final, history) in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] partial={partial!r} final={final!r}")

        seq_latency = benchmark_sequential(slm, final, history)
        sequential_latencies.append(seq_latency)

        spec_result = benchmark_speculative(
            speculator, partial, final, history,
            divergence_threshold, simulated_speaking_delay_s,
        )
        speculative_results.append(spec_result)

        print(
            f"    sequential: {seq_latency:.0f}ms | "
            f"speculative: {spec_result['effective_latency_ms']:.0f}ms "
            f"(committed={spec_result['committed']}, sim={spec_result['similarity']:.2f})"
        )

    commit_count = sum(1 for r in speculative_results if r["committed"])
    commit_rate = commit_count / len(TEST_CASES)
    spec_latencies = [r["effective_latency_ms"] for r in speculative_results]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Commit rate: {commit_rate:.1%} ({commit_count}/{len(TEST_CASES)})")
    print(f"Divergence threshold: {divergence_threshold}")
    print()
    print(f"Sequential latency  -- mean: {statistics.mean(sequential_latencies):.0f}ms | median: {statistics.median(sequential_latencies):.0f}ms")
    print(f"Speculative latency -- mean: {statistics.mean(spec_latencies):.0f}ms | median: {statistics.median(spec_latencies):.0f}ms")
    print()

    committed_latencies = [r["effective_latency_ms"] for r in speculative_results if r["committed"]]
    rolled_back_latencies = [r["effective_latency_ms"] for r in speculative_results if not r["committed"]]
    if committed_latencies:
        print(f"Committed cases only -- mean latency: {statistics.mean(committed_latencies):.0f}ms (n={len(committed_latencies)})")
    if rolled_back_latencies:
        print(f"Rolled-back cases only -- mean latency: {statistics.mean(rolled_back_latencies):.0f}ms (n={len(rolled_back_latencies)})")

    speedup = (statistics.mean(sequential_latencies) - statistics.mean(spec_latencies)) / statistics.mean(sequential_latencies)
    print(f"\nOverall speedup: {speedup:.1%} (positive = speculative is faster on average)")

    return {
        "commit_rate": commit_rate,
        "sequential_latencies": sequential_latencies,
        "speculative_results": speculative_results,
    }


if __name__ == "__main__":
    run_benchmark()
