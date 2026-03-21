import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_135M, MODEL_360M, REF_MODEL_135M, REF_MODEL_360M, TOKENIZER, DATASET


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_models_and_tokenizer(device: torch.device):
    """
    Load fine-tuned target models and pre-trained reference models, move to device.

    Returns: tokenizer, model_135m, model_360m, ref_model_135m, ref_model_360m
    """
    model_dtype = get_model_dtype(device)
    print(f"Using model dtype: {model_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    print("Loading models...")
    model_135m     = AutoModelForCausalLM.from_pretrained(MODEL_135M,     torch_dtype=model_dtype).to(device).eval()
    model_360m     = AutoModelForCausalLM.from_pretrained(MODEL_360M,     torch_dtype=model_dtype).to(device).eval()
    ref_model_135m = AutoModelForCausalLM.from_pretrained(REF_MODEL_135M, torch_dtype=model_dtype).to(device).eval()
    ref_model_360m = AutoModelForCausalLM.from_pretrained(REF_MODEL_360M, torch_dtype=model_dtype).to(device).eval()

    print("Models loaded: 135M, 360M, and their base reference models.")
    return tokenizer, model_135m, model_360m, ref_model_135m, ref_model_360m


def load_data():
    """
    Load dataset splits as DataFrames.

    Returns: dict with keys 'train', 'validation', 'test'
    """
    print("Loading dataset...")
    dataset = load_dataset(DATASET)
    splits = {
        "train":      dataset["train"].to_pandas(),
        "validation": dataset["validation"].to_pandas(),
        "test":       dataset["test"].to_pandas(),
    }
    train_df = splits["train"]
    val_df   = splits["validation"]
    print(f"Train:      {len(train_df)} samples  ({int(train_df['is_member'].sum())} members)")
    print(f"Validation: {len(val_df)} samples  ({int(val_df['is_member'].sum())} members)")
    print(f"Test:       {len(splits['test'])} samples")
    return splits
