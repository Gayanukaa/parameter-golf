"""Report parameter-golf metrics to Weights & Biases."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--artifact", default=None, help="Path to model artifact (.ptz)")
    parser.add_argument("--project", default="parameter-golf", help="W&B project name")
    parser.add_argument("--entity", default="gayanuka-lab")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping. pip install wandb")
        return

    from datetime import datetime

    metrics = json.loads(Path(args.metrics).read_text())
    exp = metrics.get("experiment", "unknown")
    seed = metrics.get("seed", 0)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"parameter-golf-{exp}-seed{seed}-{ts}"

    config = metrics.get("hyperparameters", {})
    config["experiment"] = exp
    config["seed"] = seed
    if "techniques" in metrics:
        config["techniques"] = ",".join(metrics["techniques"])

    wandb.init(project=args.project, entity=args.entity, name=run_name, config=config)

    log_data = {
        k: metrics[k] for k in [
            "val_bpb", "val_loss", "val_bpb_roundtrip", "val_loss_roundtrip",
            "val_bpb_sliding", "val_loss_sliding",
            "artifact_bytes", "code_bytes", "total_bytes",
            "training_time_ms", "steps", "step_avg_ms",
            "peak_memory_mib", "model_params",
        ] if k in metrics
    }
    wandb.log(log_data)

    if args.artifact and Path(args.artifact).exists():
        artifact = wandb.Artifact(f"{exp}-model", type="model")
        artifact.add_file(args.artifact)
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f"Logged to W&B: {args.project}/{run_name}")


if __name__ == "__main__":
    main()
