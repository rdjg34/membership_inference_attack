import io

import torch
import torch.nn.functional as F
from PIL import Image


def _load_image(raw_image) -> Image.Image:
    if isinstance(raw_image, dict) and "bytes" in raw_image:
        return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
    return raw_image.convert("RGB")


def _prepare_inputs(processor, image: Image.Image, text: str, device: torch.device):
    """
    Split text into user prompt / assistant caption, apply chat template,
    and return (inputs, prompt_len) where prompt_len is the number of
    tokens before the assistant caption begins.
    """
    parts = text.split("\n", 1)
    user_text      = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    # Full sequence: user + assistant
    messages_full = [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text",  "text": assistant_text}]},
    ]
    full_text = processor.apply_chat_template(messages_full, tokenize=False)
    inputs = processor(text=full_text, images=[image], return_tensors="pt").to(device)

    # Cast float tensors for bfloat16 devices
    if device.type == "cuda":
        inputs = {
            k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

    # Prompt-only sequence (to find where caption tokens begin)
    messages_prompt = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    return inputs, prompt_len


def compute_vlm_features(model, processor, raw_image, text: str):
    """
    Compute loss-based and token-level features for membership inference.

    Args:
        model:      Fine-tuned SmolVLM target model.
        processor:  AutoProcessor for the model.
        raw_image:  PIL Image or HuggingFace image dict.
        text:       Raw text string from the dataset (user\\nassistant format).

    Returns:
        dict of features
    """
    device = next(model.parameters()).device
    image  = _load_image(raw_image)
    inputs, prompt_len = _prepare_inputs(processor, image, text, device)

    # ── Full-sequence labels (mask padding) ──────────────────────────────────
    full_labels = inputs["input_ids"].clone()
    full_labels[full_labels == processor.tokenizer.pad_token_id] = -100

    # ── Caption-only labels (mask prompt + padding) ──────────────────────────
    caption_labels = inputs["input_ids"].clone()
    caption_labels[:, :prompt_len] = -100
    caption_labels[caption_labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        out_full    = model(**inputs, labels=full_labels)
        out_caption = model(**inputs, labels=caption_labels)

        loss         = out_full.loss.item()
        caption_loss = out_caption.loss.item()

        # Per-token probabilities (caption tokens only)
        logits       = out_full.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]

        token_probs = F.softmax(shift_logits, dim=-1)
        actual_token_probs = token_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Restrict stats to caption tokens
        cap_mask = (caption_labels[..., 1:] != -100)  # (1, seq-1)
        if cap_mask.sum() == 0:
            # fallback to full sequence if caption detection failed
            cap_mask = (full_labels[..., 1:] != -100)

        cap_probs = actual_token_probs[cap_mask]  # (n_cap_tokens,)

        mean_token_prob = cap_probs.mean().item()
        min_token_prob  = cap_probs.min().item()
        max_token_prob  = cap_probs.max().item()
        std_token_prob  = cap_probs.std().item()

        # Entropy (over full vocab at each caption position)
        cap_full_dist = token_probs[0][cap_mask[0]]          # (n_cap, vocab)
        token_entropies = -(cap_full_dist * torch.log(cap_full_dist + 1e-10)).sum(dim=-1)
        mean_entropy = token_entropies.mean().item()

        top_k_probs, _ = torch.topk(cap_full_dist, k=min(10, cap_full_dist.shape[-1]), dim=-1)
        top_k_mass = top_k_probs.sum(dim=-1).mean().item()

        low_conf_ratio = (cap_probs < 0.1).float().mean().item()

        # Min-K%
        sorted_probs = cap_probs.sort().values  # ascending
        n_tokens = sorted_probs.shape[0]
        k10 = max(1, int(n_tokens * 0.10))
        k20 = max(1, int(n_tokens * 0.20))
        k30 = max(1, int(n_tokens * 0.30))
        min_k10_prob = sorted_probs[:k10].mean().item()
        min_k20_prob = sorted_probs[:k20].mean().item()
        min_k30_prob = sorted_probs[:k30].mean().item()

        # Min-K%++
        actual_log_probs = torch.log(cap_probs + 1e-10)
        log_probs_vocab  = torch.log(cap_full_dist + 1e-10)
        mu    = log_probs_vocab.mean(dim=-1)
        sigma = log_probs_vocab.std(dim=-1)
        normalized_scores = (actual_log_probs - mu) / (sigma + 1e-10)
        sorted_normalized = normalized_scores.sort().values
        min_k10_pp = sorted_normalized[:k10].mean().item()
        min_k20_pp = sorted_normalized[:k20].mean().item()
        min_k30_pp = sorted_normalized[:k30].mean().item()

    return {
        "loss":            loss,
        "caption_loss":    caption_loss,
        "mean_token_prob": mean_token_prob,
        "min_token_prob":  min_token_prob,
        "max_token_prob":  max_token_prob,
        "std_token_prob":  std_token_prob,
        "mean_entropy":    mean_entropy,
        "top_k_mass":      top_k_mass,
        "low_conf_ratio":  low_conf_ratio,
        "min_k10_prob":    min_k10_prob,
        "min_k20_prob":    min_k20_prob,
        "min_k30_prob":    min_k30_prob,
        "min_k10_pp":      min_k10_pp,
        "min_k20_pp":      min_k20_pp,
        "min_k30_pp":      min_k30_pp,
    }
