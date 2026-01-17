# experiments/exp00_baseline/run.py
import ssl
import sys
import os
import argparse
import torch

# 1. Magic code: Import tools (works even when run from experiments subdirectory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools (keep only baseline-required)
from tools.models.clip_model import load_clip
from tools.utils.datasets import load_dataset
from tools.utils.utils import build_dataloader  # If you haven't created data_utils.py, see the "compatibility solution" below
from tools.utils.zero_shot import run_zero_shot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="eurosat",
        choices=["eurosat", "ucm", "resisc45"],
        help="Choose dataset to run zero-shot baseline.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--prompt",
        type=str,
        default="a centered satellite image of {}",
        help="Prompt template. Use {} as class placeholder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Basic setup
    ssl._create_default_https_context = ssl._create_unverified_context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 2. Local path configuration (no data download)
    dataset_root = os.path.join(project_root, "data")

    # Your weights are in weights/models/clip-vit-base-patch32
    model_path = os.path.join(project_root, "weights", "models", "clip-vit-base-patch32")

    # 3. Load model & data (both from local)
    print(f"\n[1/3] Loading CLIP from: {model_path}")
    model, processor = load_clip(model_path, device=device, use_fast=False, eval_mode=True)

    print(f"[2/3] Loading dataset: {args.dataset} from {dataset_root}")
    dataset = load_dataset(args.dataset, root=dataset_root)
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 4. Zero-shot baseline
    print(f"[3/3] Running zero-shot evaluation...")
    result = run_zero_shot(
        model=model,
        processor=processor,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
        prompt_template=args.prompt,
        logit_scale=100.0,
        desc=f"Zero-shot on {args.dataset}",
    )

    # 5. Output
    print("\n==============================")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Total samples: {result.total}")
    print(f"Zero-shot Accuracy: {result.accuracy:.4f} ({result.accuracy * 100:.2f}%)")
    print("==============================")


"""
python experiments/exp00_baseline/run.py --dataset eurosat
python experiments/exp00_baseline/run.py --dataset ucm
python experiments/exp00_baseline/run.py --dataset resisc45
"""

if __name__ == "__main__":
    main()