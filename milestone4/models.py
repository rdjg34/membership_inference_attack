import torch
from datasets import load_dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

from config import MODEL_ID, DATASET_ID


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_processor(device: torch.device):
    """
    Load the fine-tuned VLM and its processor.

    Returns: processor, model
    """
    model_dtype = get_model_dtype(device)
    print(f"Using dtype: {model_dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    print(f"Loading model: {MODEL_ID}")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
        _attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device).eval()

    print("Model loaded.")
    return processor, model


def load_data():
    """
    Load dataset splits as DataFrames.

    Returns: dict with keys 'train', 'validation', 'test'
    """
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)
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
