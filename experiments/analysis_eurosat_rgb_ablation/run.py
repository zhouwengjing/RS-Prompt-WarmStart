# experiments/analysis_eurosat_rgb_ablation/run.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# 1. Magic code: Import tools
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools (core!)
from tools.models.clip_model import LearnablePromptCLIP
from tools.utils.utils import setup_seed, get_eurosat_rgb_loader

# Configuration area
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 15  # Paper figure shows 15 epochs

# 3. Experimental configurations (Name, Use_Warm_Start, Std_Dev, Color, LineStyle)
CONFIGS = [
    ("Ours (Warm Start)", True, None, "red", "-"),
    ("Random (std=0.02)", False, 0.02, "blue", "--"),
    ("Random (std=0.1)", False, 0.10, "orange", "-."),
    ("Random (std=0.5)", False, 0.50, "green", ":")
]

# Main execution logic
def run_ablation_study():
    # 1. Lock seed (ensure data split matches Exp1)
    setup_seed(42)

    # 2. Prepare data and paths
    DATA_PATH = os.path.join(project_root, 'data')
    MODEL_NAME = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')

    print(f"Data Path: {DATA_PATH}")

    # Using unified data loader (reusing Exp1 logic)
    train_loader, test_loader, class_names = get_eurosat_rgb_loader(DATA_PATH, MODEL_NAME, BATCH_SIZE)
    if train_loader is None: return

    results = {}
    print(f"\nStarting EuroSAT 4-group comparison experiments (Total Epochs: {EPOCHS})...")

    # 3. Run 4 experimental groups
    for name, use_warm, std_val, color, style in CONFIGS:
        print(f"\n" + "=" * 60)
        print(f">>> Running Experiment: {name}")
        print("=" * 60)

        # A. Initialize model (default is Warm Start)
        model = LearnablePromptCLIP(MODEL_NAME, class_names, DEVICE)

        # B. Key modification: Overwrite ctx parameters for Random mode
        if not use_warm:
            print(f"    [Init Strategy] Overwriting with Random Noise (std={std_val})...")
            # Directly fill parameter data with normal distribution
            nn.init.normal_(model.ctx, std=std_val)
        else:
            print(f"    [Init Strategy] Keeping Warm Start ('a photo of a')...")

        # C. Optimizer
        optimizer = optim.SGD([model.ctx], lr=0.002)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        acc_history = []

        # D. Training loop
        for epoch in range(EPOCHS):
            model.train()
            # Training (skip Vision Encoder calculation for speed, or adjust based on GPU memory)
            # Note: LearnablePromptCLIP's forward already includes Vision part
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(images)  # Directly call forward
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total * 100
            acc_history.append(acc)
            print(f"    Epoch {epoch + 1}: Acc = {acc:.2f}%")

        # Record results
        duration = time.time() - start_time
        results[name] = {
            "acc": acc_history,
            "time": duration,
            "color": color,
            "style": style
        }
        print(f"Duration: {duration:.1f}s")

    # 4. Plot line chart
    plot_results(results)


def plot_results(results):
    print("\nPlotting...")
    plt.figure(figsize=(10, 7))
    epochs_range = range(1, EPOCHS + 1)

    for name, data in results.items():
        # Add time to legend label (format matches code A)
        data['time'] = data['time'] / 60
        label_str = f"{name} ({data['time']:.0f}min)"
        plt.plot(epochs_range, data['acc'], label=label_str,
                 color=data['color'], linestyle=data['style'], linewidth=2.5)

    plt.title('Effect of Initialization Dynamics on EuroSAT', fontsize=16)
    plt.xlabel('Training Epochs', fontsize=14)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save path
    save_dir = os.path.join(project_root, 'pictures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ablation_initialization.png')

    plt.savefig(save_path, dpi=300)
    print(f"Line chart saved to: {save_path}")
    # plt.show() # Comment out when running on server


if __name__ == '__main__':
    run_ablation_study()