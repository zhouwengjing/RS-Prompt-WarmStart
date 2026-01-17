# experiments/analysis_eurosat_rgb_heatmap/run.py
import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# 1. Magic code: Import tools
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools
from tools.models.clip_model import LearnablePromptCLIP
from tools.utils.utils import setup_seed, get_eurosat_rgb_loader
# [Added] Import the plotting tool we just wrote
from tools.utils.visualize import plot_confusion_matrix


def run_heatmap_analysis():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Analysis on: {device}")

    # 3. Path configuration
    model_name = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')

    # Weight path (ensure file name matches what Exp1 produced)
    weights_path = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_eurosat_seed42.pt')

    # Data should be in RS-Prompt-WarmStart/data
    data_path = os.path.join(project_root, 'data')

    # Image save path: pictures/eurosat_heatmap01.png
    save_picture_path = os.path.join(project_root, 'pictures', 'eurosat_heatmap01.png')

    # 4. Prepare data (using get_eurosat_loader to get test set)
    # Note: must use get_eurosat_loader! Because it contains random_split logic.
    # If you load the full dataset directly, the confusion matrix will be 'overly optimistic' because it includes training set data.
    _, test_loader, class_names = get_eurosat_rgb_loader(data_path, model_name, batch_size=32)

    if test_loader is None: return

    # 5. Load model and weights
    print(f"Loading weight: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"Error: weight file not found. Please run the training code in exp01_eurosat_rgb first.")
        return

    model = LearnablePromptCLIP(model_name, class_names, device)

    try:
        saved_ctx = torch.load(weights_path, map_location=device)
        model.ctx.data = saved_ctx.data
        print(">>> Weight loaded successfully!")
    except Exception as e:
        print(f" Failed to load weight: {e}")
        return

    # 6. Run inference (collect all predictions)
    model.eval()
    all_preds = []
    all_labels = []

    print("Starting inference...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 7. Plot and save
    # Calculate overall accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = np.mean(all_preds == all_labels)

    print(f"Test set accuracy: {acc * 100:.2f}%")

    # Call the plotting tool
    plot_title = f"Confusion Matrix - EuroSAT (Acc: {acc * 100:.2f}%)"
    plot_confusion_matrix(all_labels, all_preds, class_names, save_picture_path, title=plot_title)


if __name__ == '__main__':
    # Fix seed to ensure test_loader matches training data split, no data leakage
    setup_seed(42)
    run_heatmap_analysis()

# [Debug] Test Set first index: 5805