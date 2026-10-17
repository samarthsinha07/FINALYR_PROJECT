"""
model_engine.py - Core ML Inference Engine for Crisis-Fusion Dashboard
Handles model discovery, dynamic loading, caching, text/image preprocessing,
and score-level & cross-modal multimodal inference.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
    ViTModel,
    AutoModel,
)

# Optional import of CrossModalFusionModel
try:
    from fusion_model import CrossModalFusionModel
except ImportError:
    CrossModalFusionModel = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architectures and HuggingFace IDs
TEXT_BACKBONES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-large",
    "bertweet": "vinai/bertweet-base",
}

IMAGE_BACKBONES = {
    "resnet50": "torchvision/resnet50",
    "densenet121": "torchvision/densenet121",
    "vit": "google/vit-base-patch16-224-in21k",
}

# Task configuration metadata
TASK_CONFIG = {
    "task1a": {
        "title": "Task 1a — Informativeness Classification",
        "description": "Binary classification: Determines if a post contains informative disaster-related information.",
        "num_classes": 2,
        "default_text_model": "bert",
        "default_image_model": "resnet50",
        "fallback_labels": ["informative", "not_informative"],
    },
    "task1b": {
        "title": "Task 1b — Disaster Type Classification",
        "description": "Multi-class (5 classes): Categorizes the disaster into earthquake, flood, hurricane, non-informative, or wildfire.",
        "num_classes": 5,
        "default_text_model": "roberta",
        "default_image_model": "vit",
        "fallback_labels": ["earthquake", "flood", "hurricane", "non_informative", "wildfire"],
    },
    "task2": {
        "title": "Task 2 — Humanitarian Information Classification",
        "description": "Fine-grained (8 classes): Identifies affected individuals, infrastructure damage, rescue efforts, etc.",
        "num_classes": 8,
        "default_text_model": "bertweet",
        "default_image_model": "vit",
        "fallback_labels": [
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "injured_or_dead_people",
            "missing_or_found_people",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
            "vehicle_damage",
        ],
    },
    "task3": {
        "title": "Task 3 — Damage Severity Classification",
        "description": "Severity assessment (3 classes): Classifies physical destruction into little or no damage, mild damage, or severe damage.",
        "num_classes": 3,
        "default_text_model": "roberta",
        "default_image_model": "vit",
        "fallback_labels": ["little_or_no_damage", "mild_damage", "severe_damage"],
    },
}

# In-memory cache for loaded tokenizers and models
_TOKENIZER_CACHE: Dict[str, Any] = {}
_MODEL_CACHE: Dict[str, Any] = {}


def clean_tweet_text(text: str) -> str:
    """Preprocess tweet text identical to research methodology."""
    text = str(text)
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = text.replace("#", "")
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """Return canonical ImageNet evaluation transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_label_map(task: str) -> List[str]:
    """Load ordered label names from data/processed/{task}/label_map.json."""
    path = os.path.join("data", "processed", task, "label_map.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data is {"class_name": index}
            ordered = [None] * len(data)
            for k, v in data.items():
                ordered[int(v)] = str(k)
            if all(x is not None for x in ordered):
                return ordered
        except Exception:
            pass
    return TASK_CONFIG[task]["fallback_labels"]


def scan_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Scan root and figures directories for any .pt or .pth checkpoint files.
    Returns status map of all known models.
    """
    status: Dict[str, Dict[str, Any]] = {}

    for task_key, task_val in TASK_CONFIG.items():
        # Text models
        for tm in TEXT_BACKBONES.keys():
            ckpt_name = f"best_{task_key}_{tm}.pt"
            exists = os.path.exists(ckpt_name)
            size_mb = round(os.path.getsize(ckpt_name) / (1024 * 1024), 2) if exists else 0
            status[f"{task_key}_{tm}"] = {
                "task": task_key,
                "modality": "text",
                "model": tm,
                "filename": ckpt_name,
                "exists": exists,
                "size_mb": size_mb,
            }

        # Vision models
        for vm in IMAGE_BACKBONES.keys():
            ckpt_name = f"best_{task_key}_{vm}.pt"
            exists = os.path.exists(ckpt_name)
            size_mb = round(os.path.getsize(ckpt_name) / (1024 * 1024), 2) if exists else 0
            status[f"{task_key}_{vm}"] = {
                "task": task_key,
                "modality": "image",
                "model": vm,
                "filename": ckpt_name,
                "exists": exists,
                "size_mb": size_mb,
            }

        # Cross-modal fusion checkpoint
        fusion_ckpt = f"best_{task_key}_fusion.pt"
        exists = os.path.exists(fusion_ckpt)
        size_mb = round(os.path.getsize(fusion_ckpt) / (1024 * 1024), 2) if exists else 0
        status[f"{task_key}_fusion"] = {
            "task": task_key,
            "modality": "cross_modal",
            "model": "cross_modal_attention",
            "filename": fusion_ckpt,
            "exists": exists,
            "size_mb": size_mb,
        }

    return status


def get_cached_tokenizer(model_name: str):
    """Retrieve or load cached Hugging Face tokenizer."""
    if model_name not in TEXT_BACKBONES:
        raise ValueError(f"Unknown text model: {model_name}")

    hf_id = TEXT_BACKBONES[model_name]
    if hf_id not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[hf_id] = AutoTokenizer.from_pretrained(hf_id)
    return _TOKENIZER_CACHE[hf_id]


def load_text_classifier(task: str, model_name: str, ckpt_path: str):
    """Load and cache fine-tuned sequence classification model."""
    cache_key = f"{task}_{model_name}_{ckpt_path}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file '{ckpt_path}' not found on disk. "
            f"To run live inference for {model_name} on {task}, please train or place the checkpoint file in the project root."
        )

    num_classes = len(load_label_map(task))
    hf_id = TEXT_BACKBONES[model_name]

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=num_classes,
    )
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def load_image_classifier(task: str, model_name: str, ckpt_path: str):
    """Load and cache fine-tuned vision classification model."""
    cache_key = f"{task}_{model_name}_{ckpt_path}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file '{ckpt_path}' not found on disk. "
            f"To run live image inference for {model_name} on {task}, please train or place the checkpoint file in the project root."
        )

    num_classes = len(load_label_map(task))

    if model_name == "vit":
        model = ViTForImageClassification.from_pretrained(
            IMAGE_BACKBONES["vit"],
            num_labels=num_classes,
        )
    elif model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported vision model: {model_name}")

    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def load_cross_modal_model(task: str, ckpt_path: str):
    """Load and cache cross-modal attention fusion model."""
    if CrossModalFusionModel is None:
        raise ImportError("fusion_model.py is not available in workspace.")

    cache_key = f"{task}_crossmodal_{ckpt_path}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Cross-modal checkpoint file '{ckpt_path}' not found on disk."
        )

    num_classes = len(load_label_map(task))
    model = CrossModalFusionModel(num_classes=num_classes)

    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def predict_text(task: str, model_name: str, text: str, ckpt_path: Optional[str] = None) -> Dict[str, float]:
    """Execute forward pass for text modality."""
    if not ckpt_path:
        ckpt_path = f"best_{task}_{model_name}.pt"

    labels = load_label_map(task)
    tokenizer = get_cached_tokenizer(model_name)
    model = load_text_classifier(task, model_name, ckpt_path)

    cleaned = clean_tweet_text(text)
    encoded = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(labels)}


def predict_image(task: str, model_name: str, image: Image.Image, ckpt_path: Optional[str] = None) -> Dict[str, float]:
    """Execute forward pass for visual modality."""
    if not ckpt_path:
        ckpt_path = f"best_{task}_{model_name}.pt"

    labels = load_label_map(task)
    model = load_image_classifier(task, model_name, ckpt_path)

    tf = get_image_transform()
    img_rgb = image.convert("RGB")
    tensor = tf(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "logits"):
            logits = model(tensor).logits
        elif model_name == "vit":
            logits = model(pixel_values=tensor).logits
        else:
            logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(labels)}


def predict_score_fusion(
    task: str,
    text_model_name: str,
    image_model_name: str,
    text: str,
    image: Image.Image,
    weight: float = 0.5,
    text_ckpt: Optional[str] = None,
    image_ckpt: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Score-level fusion: combines text and image probability vectors.
    P_fused = weight * P_text + (1 - weight) * P_image
    Returns (fused_probs, text_probs, image_probs).
    """
    text_probs = predict_text(task, text_model_name, text, text_ckpt)
    image_probs = predict_image(task, image_model_name, image, image_ckpt)

    labels = load_label_map(task)
    fused_probs = {}
    for label in labels:
        p_t = text_probs.get(label, 0.0)
        p_i = image_probs.get(label, 0.0)
        fused_probs[label] = float(weight * p_t + (1.0 - weight) * p_i)

    return fused_probs, text_probs, image_probs


def predict_cross_modal(
    task: str,
    text: str,
    image: Image.Image,
    ckpt_path: Optional[str] = None,
) -> Dict[str, float]:
    """Execute forward pass on cross-modal attention fusion architecture."""
    if not ckpt_path:
        ckpt_path = f"best_{task}_fusion.pt"

    labels = load_label_map(task)
    tokenizer = get_cached_tokenizer("roberta")
    model = load_cross_modal_model(task, ckpt_path)

    cleaned = clean_tweet_text(text)
    encoded = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    tf = get_image_transform()
    img_rgb = image.convert("RGB")
    img_tensor = tf(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=img_tensor,
        )
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(labels)}


def load_dataset_samples(task: str, split: str = "test", num_samples: int = 5) -> List[Dict[str, Any]]:
    """Fetch real paired samples from the processed CrisisMMD dataset."""
    csv_path = os.path.join("data", "processed", task, f"{split}.csv")
    if not os.path.exists(csv_path):
        return []

    try:
        df = pd.read_csv(csv_path)
        samples = []
        for _, row in df.head(num_samples).iterrows():
            samples.append({
                "text": str(row["text"]),
                "image_path": str(row["image_path"]),
                "label_name": str(row["label_name"]),
                "label": int(row["label"]),
            })
        return samples
    except Exception:
        return []
