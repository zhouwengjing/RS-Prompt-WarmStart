# experiments/exp03_resisc45/run.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1. Magic code: Import tools
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools
from tools.models.clip_model import LearnablePromptCLIP
from tools.utils.utils import setup_seed, get_resisc_loader, evaluate, evaluate_detailed_accuracy, parse_args

# Mode A: Training workflow
def run_resisc_training(device):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Path configuration
    model_name = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')
    # Dataset path (Note the folder name NWPU_RESISC45)
    DATASET_PATH = os.path.join(project_root, 'data', 'NWPU_RESISC45')

    # Output path
    output_dir = os.path.join(project_root, 'weights', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'best_prompt_resisc45_seed42.pt')

    print(f"Model Path: {model_name}")
    print(f"Data Path: {DATASET_PATH}")

    # 1. Load data
    # Set BATCH_SIZE to 32, change to 16 if OOM
    train_loader, test_loader, class_names = get_resisc_loader(DATASET_PATH, model_name, batch_size=32)
    if train_loader is None: return

    print("\n" + "=" * 50)
    print("Experiment C: Generalization (Training CoOp on RESISC45)")
    print("=" * 50)

    # 2. Initialize model
    model_new = LearnablePromptCLIP(model_name, class_names, device)
    optimizer = optim.SGD([model_new.ctx], lr=0.002)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 25

    best_acc = 0.0

    # 3. Training loop
    for epoch in range(EPOCHS):
        model_new.train()
        total_loss = 0

        # Add try-except to prevent bad images from interrupting training
        try:
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model_new(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        except Exception as e:
            print(f"⚠Warning: Error during training (skipped batch): {e}")
            continue

        # Simple validation
        acc = evaluate(model_new, test_loader, device, desc=f"Epoch {epoch + 1} Val")

        if acc > best_acc:
            best_acc = acc
            print(f"New Best! Saving parameters to {save_path} ...")
            torch.save(model_new.ctx, save_path)

        print(f"Epoch {epoch + 1}: Val Acc = {acc:.4f} (Best: {best_acc:.4f})")

    print(f"\n[Training completed] Best Accuracy on RESISC45: {best_acc * 100:.2f}%")

    # 4. Detailed evaluation
    print("\nLoading best weights for final detailed evaluation...")
    if os.path.exists(save_path):
        best_ctx = torch.load(save_path, map_location=device)
        model_new.ctx.data = best_ctx.data
        evaluate_detailed_accuracy(model_new, test_loader, device, class_names, title="Final Result on RESISC45")
    else:
        print("Error: Could not find best weights file.")

# Mode B: Validation workflow
def run_validation(device):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Validation Mode Running on: {device}")

    model_name = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')
    DATASET_PATH = os.path.join(project_root, 'data', 'NWPU_RESISC45')
    param_path = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_resisc45_seed42.pt')

    # 1. Prepare data
    _, test_loader, class_names = get_resisc_loader(DATASET_PATH, model_name, batch_size=32)
    if test_loader is None: return

    # 2. Initialize model
    model = LearnablePromptCLIP(model_name, class_names, device)

    # 3. Load weights
    if os.path.exists(param_path):
        print(f"Loading parameter file: {param_path}")
        try:
            saved_ctx = torch.load(param_path, map_location=device)
            model.ctx.data = saved_ctx.data
            print(">>> Parameter loading successful!")

            evaluate_detailed_accuracy(model, test_loader, device, class_names, title="Validation Mode Result")
        except Exception as e:
            print(f"Loading failed: {e}")
    else:
        print(f"Error: Weight file {param_path} not found, please run training mode first.")


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
        run_resisc_training(DEVICE)
    elif args.mode == "validate":
        run_validation(DEVICE)
    else:
        print("Unknown error!")