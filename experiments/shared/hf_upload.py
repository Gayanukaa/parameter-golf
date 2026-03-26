"""Upload parameter-golf model and metrics to HuggingFace Hub."""

import argparse
import json
import os
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to compressed model (.ptz)")
    parser.add_argument("--script", required=True, help="Path to train_gpt.py")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON")
    parser.add_argument("--repo", required=True, help="HF repo owner (e.g. Gayanukaa)")
    parser.add_argument("--run-name", default=None, help="Run name (auto-generated if not set)")
    parser.add_argument("--token", default=None, help="HF token (or set HUGGINGFACE_TOKEN env)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. pip install huggingface-hub")
        return

    token = args.token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("No HF token found. Set HUGGINGFACE_TOKEN or pass --token")
        return

    api = HfApi(token=token)

    # Build repo name with naming convention
    from datetime import datetime
    metrics = json.loads(Path(args.metrics).read_text())
    exp = metrics.get("experiment", "unknown")
    seed = metrics.get("seed", 0)
    if args.run_name:
        repo_name = args.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        repo_name = f"parameter-golf-{exp}-seed{seed}-{ts}"
    repo_id = f"{args.repo}/{repo_name}"

    # Create repo if needed
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"Repo creation: {e}")

    # Create zip bundle
    model_path = Path(args.model)
    script_path = Path(args.script)
    metrics_path = Path(args.metrics)
    zip_path = model_path.parent / f"{model_path.stem}_bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if model_path.exists():
            zf.write(model_path, model_path.name)
        if script_path.exists():
            zf.write(script_path, script_path.name)
        if metrics_path.exists():
            zf.write(metrics_path, metrics_path.name)

    # Upload files
    files_to_upload = [
        (model_path, model_path.name),
        (script_path, "train_gpt.py"),
        (metrics_path, "metrics.json"),
        (zip_path, zip_path.name),
    ]

    for local, remote in files_to_upload:
        if local.exists():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
            )
            print(f"Uploaded: {remote}")

    # Create model card
    card = f"""---
tags: [parameter-golf, language-model, compression]
---
# {repo_name}

Experiment: **{metrics.get('experiment', 'unknown')}** | Seed: {metrics.get('seed', 'N/A')}

| Metric | Value |
|--------|-------|
| BPB | {metrics.get('val_bpb', 'N/A')} |
| BPB (roundtrip) | {metrics.get('val_bpb_roundtrip', 'N/A')} |
| Artifact | {metrics.get('total_bytes', 'N/A')} bytes |
| Steps | {metrics.get('steps', 'N/A')} |
| Train time | {metrics.get('training_time_ms', 'N/A')}ms |

Download the zip bundle for all files.
"""
    readme_path = model_path.parent / "README_hf.md"
    readme_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo,
    )
    readme_path.unlink()

    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
