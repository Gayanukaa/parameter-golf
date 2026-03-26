"""Parse parameter-golf training logs into structured JSON metrics."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_log(log_path: str) -> dict:
    text = Path(log_path).read_text()
    m = {}

    # Final metrics (prefer TTT > sliding window > standard roundtrip)
    for pattern, prefix in [
        (r"legal_ttt_exact val_loss:([\d.]+) val_bpb:([\d.]+)", "ttt"),
        (r"final_int6_sliding_window_exact val_loss:([\d.]+) val_bpb:([\d.]+)", "sliding"),
        (r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", "roundtrip"),
        (r"final_int6_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", "roundtrip"),
    ]:
        match = re.search(pattern, text)
        if match:
            key = f"val_loss_{prefix}" if prefix != "ttt" else "val_loss"
            bpb_key = f"val_bpb_{prefix}" if prefix != "ttt" else "val_bpb"
            m[key] = float(match.group(1))
            m[bpb_key] = float(match.group(2))

    # Use best available as primary metric
    if "val_bpb" not in m:
        for k in ["val_bpb_sliding", "val_bpb_roundtrip"]:
            if k in m:
                m["val_bpb"] = m[k]
                m["val_loss"] = m[k.replace("bpb", "loss")]
                break

    # Roundtrip metrics (always capture if present)
    for pattern in [
        r"final_int6_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)",
        r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)",
    ]:
        match = re.search(pattern, text)
        if match:
            m["val_bpb_roundtrip"] = float(match.group(2))
            m["val_loss_roundtrip"] = float(match.group(1))
            break

    # Artifact size
    for pattern in [
        r"Serialized model int6\+lzma: (\d+) bytes",
        r"Serialized model int6\+zstd: (\d+) bytes",
        r"Serialized model int8\+zlib: (\d+) bytes",
        r"Serialized model int8\+zstd: (\d+) bytes",
    ]:
        match = re.search(pattern, text)
        if match:
            m["artifact_bytes"] = int(match.group(1))
            break

    match = re.search(r"Code size: (\d+) bytes", text)
    if match:
        m["code_bytes"] = int(match.group(1))

    for pattern in [
        r"Total submission size int6\+lzma: (\d+) bytes",
        r"Total submission size int6\+zstd: (\d+) bytes",
        r"Total submission size int8\+zlib: (\d+) bytes",
        r"Total submission size int8\+zstd: (\d+) bytes",
    ]:
        match = re.search(pattern, text)
        if match:
            m["total_bytes"] = int(match.group(1))
            break

    # Training stats
    match = re.search(r"model_params:(\d+)", text)
    if match:
        m["model_params"] = int(match.group(1))

    match = re.search(r"peak memory allocated: (\d+) MiB", text)
    if match:
        m["peak_memory_mib"] = int(match.group(1))

    # Last training step before stopping
    steps = re.findall(r"step:(\d+)/\d+ (?:train_loss|val_loss):.+ train_time:(\d+)ms step_avg:([\d.]+)ms", text)
    if steps:
        last = steps[-1]
        m["steps"] = int(last[0])
        m["training_time_ms"] = int(last[1])
        m["step_avg_ms"] = float(last[2])

    # Hyperparameters from log header
    hp = {}
    hp_patterns = {
        "num_layers": r"num_layers[:\s=]+(\d+)",
        "model_dim": r"model_dim[:\s=]+(\d+)",
        "vocab_size": r"VOCAB_SIZE=(\d+)|vocab_size[:\s=]+(\d+)",
        "train_seq_len": r"train_seq_len:(\d+)",
        "train_batch_tokens": r"train_batch_tokens:(\d+)",
        "warmdown_iters": r"warmdown_iters[:\s=]+(\d+)",
        "matrix_lr": r"matrix_lr:([\d.]+)",
        "embed_lr": r"embed_lr:([\d.]+)",
        "seed": r"seed:(\d+)",
        "iterations": r"iterations:(\d+)",
    }
    for key, pat in hp_patterns.items():
        match = re.search(pat, text)
        if match:
            val = next(g for g in match.groups() if g is not None)
            hp[key] = float(val) if "." in val else int(val)
    if hp:
        m["hyperparameters"] = hp

    # Seed from hyperparameters
    if "seed" not in m and "seed" in hp:
        m["seed"] = hp["seed"]

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default=".")
    parser.add_argument("--techniques", nargs="*", default=[])
    args = parser.parse_args()

    metrics = parse_log(args.log)
    metrics["experiment"] = args.experiment
    metrics["seed"] = args.seed
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    if args.techniques:
        metrics["techniques"] = args.techniques

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.experiment}_seed{args.seed}_{ts}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_path}")

    # Print summary
    bpb = metrics.get("val_bpb", "N/A")
    total = metrics.get("total_bytes", "N/A")
    steps = metrics.get("steps", "N/A")
    print(f"  BPB: {bpb}  Artifact: {total}  Steps: {steps}")


if __name__ == "__main__":
    main()
