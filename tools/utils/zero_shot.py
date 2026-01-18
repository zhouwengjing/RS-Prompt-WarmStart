# tools/utils/zero_shot.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import torch
from tqdm import tqdm


@dataclass
class ZeroShotResult:
    accuracy: float
    correct: int
    total: int
    prompts: List[str]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def build_prompts(class_names: Sequence[str], template: str) -> List[str]:
    return [template.format(c) for c in class_names]


@dataclass
def build_text_features(model, processor, prompts: Sequence[str], device) -> torch.Tensor:
    """
    Precomputed text features：shape [num_classes, dim]
    """
    text_inputs = processor(text=list(prompts), return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    return _l2_normalize(text_features)


@torch.no_grad()
def zero_shot_eval(
    model,
    processor,
    dataloader,
    text_features: torch.Tensor,
    device,
    logit_scale: float = 100.0,
    desc: str = "Evaluating",
):
    """
    return: (acc, correct, total)
    """
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=desc):
        labels = labels.to(device)

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = _l2_normalize(image_features)

        logits = logit_scale * image_features @ text_features.T
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / max(total, 1)
    return acc, correct, total


def run_zero_shot(
    *,
    model,
    processor,
    dataset,
    dataloader,
    device,
    # prompt_template: str = "a centered satellite image of {}",
    prompt_template: str = "a photo of a {}",
    logit_scale: float = 100.0,
    desc: str = "Evaluating",
):
    """
    An end-to-end zero-shot gateway: Given a dataset and dataloader, it directly outputs results.
    """
    class_names = dataset.classes
    prompts = build_prompts(class_names, prompt_template)
    text_features = build_text_features(model, processor, prompts, device)

    acc, correct, total = zero_shot_eval(
        model=model,
        processor=processor,
        dataloader=dataloader,
        text_features=text_features,
        device=device,
        logit_scale=logit_scale,
        desc=desc,
    )

    return ZeroShotResult(
        accuracy=acc,
        correct=correct,
        total=total,
        prompts=prompts,
    )
