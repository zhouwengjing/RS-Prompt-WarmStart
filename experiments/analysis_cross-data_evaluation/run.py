# experiments/analysis_cross-data_evaluation/run.py
import os
import torch
import sys
from tqdm import tqdm


# 1. Magic code: Import tools
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 2. Import from tools
from tools.models.clip_model import LearnablePromptCLIP
# Import all dataset loaders
from tools.utils.utils import setup_seed, get_ucm_loader, get_resisc_loader, evaluate_detailed_accuracy

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# "Source" weights (trained on EuroSAT)
EUROSAT_WEIGHT_PATH = os.path.join(project_root, 'weights', 'outputs', 'best_prompt_eurosat_seed42.pt')

# Base model path
MODEL_NAME = os.path.join(project_root, 'weights', 'models', 'clip-vit-base-patch32')

# Dataset root directory
DATA_ROOT = os.path.join(project_root, 'data')  # All data resides here


def run_transfer_test(target_dataset_name):
    print("\n" + "#" * 60)
    print(f" Transfer Test: Using [EuroSAT] weights -> Testing [{target_dataset_name}]")
    print("#" * 60)

    # 1. Prepare target dataset
    if target_dataset_name == 'UCM':
        data_path = os.path.join(DATA_ROOT, 'UCMerced_LandUse', 'Images')
        # Get test set (Note: we only care about test_loader here)
        _, test_loader, class_names = get_ucm_loader(data_path, MODEL_NAME, BATCH_SIZE)

    elif target_dataset_name == 'RESISC45':
        data_path = os.path.join(DATA_ROOT, 'NWPU_RESISC45')
        _, test_loader, class_names = get_resisc_loader(data_path, MODEL_NAME, BATCH_SIZE)

    else:
        print("Unknown dataset")
        return

    if test_loader is None:
        print(f"Failed to load {target_dataset_name} data, please check paths.")
        return

    # 2. Initialize model (Note: using target dataset's class names!)
    # Critical step: Model structure must match target dataset's class count (UCM=21, RESISC=45)
    model = LearnablePromptCLIP(MODEL_NAME, class_names, DEVICE)

    # 3. Core step: Force-inject EuroSAT weights
    if os.path.exists(EUROSAT_WEIGHT_PATH):
        print(f"Loading EuroSAT weights: {EUROSAT_WEIGHT_PATH}")
        eurosat_ctx = torch.load(EUROSAT_WEIGHT_PATH, map_location=DEVICE)

        # Verify shape compatibility (context length usually matches, so direct assignment works)
        # model.ctx shape is [16, 512]
        if model.ctx.shape == eurosat_ctx.shape:
            model.ctx.data = eurosat_ctx.data
            print(">>> Transfer successful! EuroSAT Prompt has been implanted into the model.")
        else:
            print(f"Shape mismatch! Model: {model.ctx.shape}, Weights: {eurosat_ctx.shape}")
            return
    else:
        print(f"Error: EuroSAT weight file not found, please complete Experiment 1 first.")
        return

    # 4. Evaluation
    print(f"Starting evaluation on {target_dataset_name}...")
    evaluate_detailed_accuracy(model, test_loader, DEVICE, class_names,
                               title=f"Transfer Result (EuroSAT -> {target_dataset_name})")


if __name__ == '__main__':
    setup_seed(42)

    # 1. Test transfer to UCM
    run_transfer_test('UCM')

    # 2. Test transfer to RESISC45
    run_transfer_test('RESISC45')