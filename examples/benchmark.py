"""Benchmark one or more models on labelled examples, broken down by task.

    python examples/benchmark.py qwen2.5:7b qwen2.5:14b
    python examples/benchmark.py gpt-4.1-mini --backend openai
    python examples/benchmark.py my-model --data my_examples.jsonl

Each line of the data file is a decision plus its "answer"; see verdict/dashboard/benchmark.jsonl
(also loadable from the dashboard: Evaluate -> "Load the 36-example benchmark").
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from verdict import Verdict, evaluate

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--base-url", help="Ollama host or OpenAI-compatible /v1 URL")
    ap.add_argument("--data", default=str(HERE.parent / "verdict" / "dashboard" / "benchmark.jsonl"))
    args = ap.parse_args()
    examples = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]

    for model in args.models:
        cfg = {"model": model}
        if args.base_url:
            cfg["host" if args.backend == "ollama" else "base_url"] = args.base_url
        v = Verdict.from_backend(args.backend, **cfg)
        v.decide({"question": "warm-up", "kind": "binary"})  # load the model so latency is fair
        tasks: dict[str, list] = defaultdict(list)

        def record(d, r, answer, tasks=tasks):
            tasks[d.question].append((r.choice == answer, r.coverage))

        out = evaluate(v, examples, on_result=record)
        v.close()
        print(f"\n## {model}  ({out['n']} examples, {out['error_count']} errors, {out['avg_latency_ms']:.0f} ms per question)")
        print(f"{'task':42s} {'right':>7s} {'coverage':>9s}")
        for question, rows in tasks.items():
            right = sum(ok for ok, _ in rows)
            covs = [c for _, c in rows if c is not None]
            cov = f"{statistics.mean(covs):.0%}" if covs else "n/a"
            print(f"{question[:42]:42s} {right:>3d}/{len(rows):<3d} {cov:>9s}")
        raw, cal = out["raw"], out["calibrated"]
        print(f"accuracy {raw['accuracy']:.1%} | log loss {raw['log_loss']} -> {cal['log_loss']} calibrated | "
              f"ECE {raw['ece']} -> {cal['ece']} | fitted temperature {out['fitted_temperature']}"
              + ("" if out.get("reliable", True) else f"  (not reliable: {out['warning']})"))


if __name__ == "__main__":
    main()
