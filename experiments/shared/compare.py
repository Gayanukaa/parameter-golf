"""Compare all parameter-golf experiment results."""

import json
import sys
from pathlib import Path


def main():
    metrics_dir = Path(__file__).parent.parent / "metrics"
    if not metrics_dir.exists():
        print("No metrics/ directory found.")
        return

    files = sorted(metrics_dir.glob("*.json"))
    if not files:
        print("No metrics files found.")
        return

    results = []
    for f in files:
        try:
            m = json.loads(f.read_text())
            results.append(m)
        except (json.JSONDecodeError, KeyError):
            continue

    results.sort(key=lambda x: x.get("val_bpb", 999))

    header = f"{'Experiment':<25} {'Seed':>5} {'BPB':>8} {'BPB(RT)':>8} {'Artifact':>10} {'Steps':>6} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for m in results:
        exp = m.get("experiment", "?")[:24]
        seed = m.get("seed", "?")
        bpb = m.get("val_bpb", 0)
        bpb_rt = m.get("val_bpb_roundtrip", m.get("val_bpb_sliding", 0))
        artifact = m.get("total_bytes", 0)
        steps = m.get("steps", 0)
        time_s = m.get("training_time_ms", 0) / 1000

        artifact_str = f"{artifact / 1e6:.2f}MB" if artifact else "N/A"
        print(f"{exp:<25} {seed:>5} {bpb:>8.4f} {bpb_rt:>8.4f} {artifact_str:>10} {steps:>6} {time_s:>6.0f}s")

    # Summary stats per experiment
    exps = {}
    for m in results:
        exp = m.get("experiment", "?")
        exps.setdefault(exp, []).append(m.get("val_bpb", 999))

    if any(len(v) > 1 for v in exps.values()):
        print(f"\n{'Experiment':<25} {'Mean':>8} {'Std':>8} {'Seeds':>6}")
        print("-" * 50)
        for exp, bpbs in sorted(exps.items(), key=lambda x: sum(x[1]) / len(x[1])):
            mean = sum(bpbs) / len(bpbs)
            std = (sum((b - mean) ** 2 for b in bpbs) / len(bpbs)) ** 0.5
            print(f"{exp:<25} {mean:>8.4f} {std:>8.4f} {len(bpbs):>6}")


if __name__ == "__main__":
    main()
