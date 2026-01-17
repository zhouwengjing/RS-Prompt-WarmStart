# tools/utils/common.py
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from transformers import CLIPProcessor
from tqdm import tqdm
from torchvision.datasets import EuroSAT
import argparse
from typing import List, Tuple, Any
import torch
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Utility functions (with path checking)

# [New] EuroSAT data loader (modified from run.py)
def get_eurosat_rgb_loader(root_path, model_name, batch_size=32):
    print(f"Preparing EuroSAT data, path: {root_path}")

    try:
        dataset = EuroSAT(root=root_path, download=True, transform=None)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return None, None, None

    class_names = dataset.classes
    print(f"Classes: {class_names}")

    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(model_name)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'], torch.tensor(labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, test_loader, class_names

def get_ucm_loader(root_path, model_name, batch_size=32):
    if not os.path.exists(root_path):
        print(f"Error: UCMerced dataset path not found: {root_path}")
        return None, None, None

    print(f"Loading processor from: {model_name}")
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error: Processor loading failed. Please check model path. {e}")
        return None, None, None

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'], torch.tensor(labels)

    dataset = ImageFolder(root=root_path, transform=None)
    print(f"UCMerced loaded: {len(dataset)} images, {len(dataset.classes)} classes")

    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, dataset.classes

# [New] RESISC45 data loader
def get_resisc_loader(dataset_path, model_name, batch_size=32):
    print(f"Preparing RESISC45 data: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return None, None, None

    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(model_name)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'], torch.tensor(labels)

    try:
        dataset = ImageFolder(root=dataset_path, transform=None)
    except Exception as e:
        print(f"Loading failed: {e}")
        return None, None, None

    print(f"RESISC45 loaded: {len(dataset)} images, {len(dataset.classes)} classes")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return train_loader, test_loader, dataset.classes

def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# [New] Detailed evaluation function (for all experiments!)
def evaluate_detailed_accuracy(model, dataloader, device, class_names, title="Evaluation"):
    print(f"\n{'-' * 20} {title} {'-' * 20}")
    model.eval()
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Calculating details"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            c = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f"\n{'Class Name':<25} | {'Accuracy':<10} | {'Count'}")
    print(f"{'-' * 25}-+-{'-' * 10}-+-{'-' * 10}")

    total_correct = 0
    total_samples = 0
    for i in range(len(class_names)):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{class_names[i]:<25} | {acc:6.2f}%    | {int(class_correct[i])}/{int(class_total[i])}")
        total_correct += class_correct[i]
        total_samples += class_total[i]

    avg_acc = 100 * total_correct / total_samples
    print(f"{'-' * 50}")
    print(f"Overall Accuracy: {avg_acc:.2f}%")
    print(f"{'-' * 50}\n")
    return avg_acc

def pil_collate_fn(batch: List[Tuple[Any, int]]):
    """
    batch: [(PIL.Image, label), ...]
    return:
      images: [PIL.Image, ...]
      labels: Tensor([..])
    """
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def build_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    collate_fn=pil_collate_fn,
):
    """
    Unified DataLoader builder.
    Windows: If encounter multiprocessing/serialization issues, set num_workers to 0.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Model training and validation script")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "validate", "test"],
                        help="Running mode: train or validate")

    parser.add_argument("--device", type=str, default=None,
                        help="Specify device: cuda, cpu or cuda:0, default auto-select")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path, default use predefined CLIP model")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, default 42")

    return parser.parse_args()