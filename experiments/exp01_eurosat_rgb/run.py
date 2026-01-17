# experiments/exp00_eurosat_rgb/run.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# 1. Magic code: Import tools
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools
from tools.models.clip_model import LearnablePromptCLIP
from tools.utils.utils import setup_seed, get_eurosat_rgb_loader, evaluate_detailed_accuracy, parse_args


# Mode A: Training workflow
def run_training(model_name, device):
    BATCH_SIZE = 32
    LR = 0.002
    EPOCHS = 15

    # [Core fix] Define data path (points to root data directory)
    # Your data should be in RS-Prompt-WarmStart/data
    DATA_PATH = os.path.join(project_root, 'data')

    # Weight save path
    save_path = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_eurosat_seed42.pt')

    print(f"Data Path: {DATA_PATH}")

    # 1. Get data (pass DATA_PATH!)
    train_loader, test_loader, class_names = get_eurosat_rgb_loader(DATA_PATH, model_name, BATCH_SIZE)
    if train_loader is None: return

    # 2. Initialize model
    print("Initializing CoOp model...")
    model = LearnablePromptCLIP(model_name, class_names, device)

    # 3. Optimizer
    optimizer = optim.SGD([model.ctx], lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4. Evaluate before training
    # evaluate_detailed_accuracy(model, test_loader, device, class_names, title="Baseline (Zero-shot / Warm Start)")

    print(f"\nStarting training (Total Epochs: {EPOCHS})...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Simple validation
        model.eval()
        correct = 0;
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New record! Saving to {save_path} ...")
            torch.save(model.ctx, save_path)

    print(f"\nTraining completed. Best accuracy: {best_acc:.4f}")


# Mode B: Validation workflow
def run_validation(model_name, device):
    save_path = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_eurosat_seed42.pt')

    # [Core fix] Same path to root data
    DATA_PATH = os.path.join(project_root, 'data')

    # 1. Get data
    _, test_loader, class_names = get_eurosat_rgb_loader(DATA_PATH, model_name, batch_size=32)

    # 2. Initialize model
    model = LearnablePromptCLIP(model_name, class_names, device)

    # 3. Load weights
    if os.path.exists(save_path):
        print(f"Loading parameter file: {save_path}")
        saved_ctx = torch.load(save_path, map_location=device)
        model.ctx.data = saved_ctx.data

        # 4. Use detailed evaluation from tools
        evaluate_detailed_accuracy(model, test_loader, device, class_names, title="Validation Mode Result")
    else:
        print(f"Error: Weight file {save_path} not found")


"""
python run.py --mode train  # Training mode
python run.py --mode validate  # Validation mode
python run.py --mode train --device cuda:0 --seed 42
python run.py --help
"""

if __name__ == '__main__':
    args = parse_args()

    # Use command-line arguments to override defaults
    DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = args.model_path if args.model_path else os.path.join(
        project_root, 'weights', 'models', 'clip-vit-base-patch32'
    )

    setup_seed(args.seed)  # Use seed specified in command line

    if args.mode == "train":
        run_training(MODEL_NAME, DEVICE)
    elif args.mode == "validate":
        run_validation(MODEL_NAME, DEVICE)
    else:
        print("Unknown error!")