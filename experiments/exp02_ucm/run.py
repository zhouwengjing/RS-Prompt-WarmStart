# experiments/exp02_ucm/run.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1. Magic code: Import tools (standard approach)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming run.py is in experiments/exp02_ucm/, root directory is two levels up
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools (enjoying engineering convenience!)
from tools.models.clip_model import LearnablePromptCLIP
# Import evaluate_detailed_accuracy
from tools.utils.utils import setup_seed, get_ucm_loader, evaluate, evaluate_detailed_accuracy, parse_args


# Mode A: Training workflow
def run_ucm_training(device):
    print(f"Running on: {device}")

    # Path configuration
    model_name = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')
    UCM_PATH = os.path.join(project_root, 'data', 'UCMerced_LandUse', 'Images')

    # Output path: weights/outputs/ (plural)
    output_dir = os.path.join(project_root, 'weights', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'best_prompt_ucm42.pt')

    print(f"Model Path: {model_name}")
    print(f"Data Path: {UCM_PATH}")

    # 1. Load data
    train_loader, test_loader, class_names = get_ucm_loader(UCM_PATH, model_name)
    if train_loader is None: return

    print("\n" + "=" * 50)
    print("Experiment B: Generalization (Training CoOp on UCMerced)")
    print("=" * 50)

    # 2. Initialize model
    model_new = LearnablePromptCLIP(model_name, class_names, device)
    optimizer = optim.SGD([model_new.ctx], lr=0.002)
    criterion = nn.CrossEntropyLoss()
    # Can set higher, observe accuracy per training, peaks at 18 epochs
    EPOCHS = 18

    best_acc = 0.0

    # 3. Training loop
    for epoch in range(EPOCHS):
        model_new.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model_new(images), labels)
            loss.backward()
            optimizer.step()

        # Only track overall accuracy during training for clarity
        acc = evaluate(model_new, test_loader, device, desc=f"Epoch {epoch + 1} Val")

        if acc > best_acc:
            best_acc = acc
            print(f"New Best! Saving parameters to {save_path} ...")
            torch.save(model_new.ctx, save_path)

        print(f"Epoch {epoch + 1}: Val Acc = {acc:.4f} (Best: {best_acc:.4f})")

    print(f"\n[Training completed] Best Accuracy on UCM: {best_acc * 100:.2f}%")

    # 4. Post-training highlight: Load best weights for detailed evaluation
    print("\nLoading best weights for final detailed evaluation...")
    if os.path.exists(save_path):
        best_ctx = torch.load(save_path, map_location=device)
        model_new.ctx.data = best_ctx.data

        # Call advanced evaluation function from tools
        evaluate_detailed_accuracy(model_new, test_loader, device, class_names, title="Final Result on UCMerced")
    else:
        print("Error: Could not find best weights file.")

# Mode B: Validation workflow (reproduction)
def run_validation(device):
    print(f"Validation Mode Running on: {device}")

    # Path configuration
    model_name = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')
    UCM_PATH = os.path.join(project_root, 'data', 'UCMerced_LandUse', 'Images')
    # Ensure loading from outputs folder
    param_path = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_ucm42.pt')

    # 1. Prepare data (reuse tools)
    # Since seed is fixed, test_loader matches training exactly
    _, test_loader, class_names = get_ucm_loader(UCM_PATH, model_name)
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

            # 4. Run detailed evaluation
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
        run_ucm_training(DEVICE)
    elif args.mode == "validate":
        run_validation(DEVICE)
    else:
        print("Unknown error!")