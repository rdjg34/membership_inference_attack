"""
PC-MMIA Feature Extraction
Zheng et al., EMNLP Findings 2025 — "Tracing Training Footprints"

Core idea:
  For each caption token, measure how much its log-probability changes
  when small Gaussian noise is added to the image embeddings.
  Tokens whose probability drops significantly under perturbation are
  likely memorized from training (not just easy to predict from context).
  These tokens get higher weights in the final membership score.

Features extracted per sample:
  pcmmia_score          — weighted sum of token log-probs (primary MIA signal)
  pcmmia_mean_delta     — mean per-token sensitivity to image perturbation
  pcmmia_max_delta      — maximum per-token sensitivity
  pcmmia_std_delta      — std of per-token sensitivity
  pcmmia_top10_delta    — mean delta for top 10% most sensitive tokens
  pcmmia_top25_delta    — mean delta for top 25% most sensitive tokens

Code written with reliance on Claude AI (Anthropic, 2025).
"""

import io
import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ── Noise sigma — controls perturbation strength ───────────────────────────────
# 0.1 is the value used in the PC-MMIA paper.
# Too small: perturbation too weak to detect memorization signal
# Too large: destroys image semantics, signal becomes noise
SIGMA = 0.1


def _load_image(raw_image) -> Image.Image:
    if isinstance(raw_image, dict) and "bytes" in raw_image:
        return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
    return raw_image.convert("RGB")


def _prepare_inputs(processor, image: Image.Image, text: str, device, dtype):
    """
    Prepare tokenized inputs and compute prompt length.
    Returns (inputs_dict, prompt_len, cap_mask).
    """
    parts          = text.split("\n", 1)
    user_text      = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    messages_full = [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text",  "text": assistant_text}]},
    ]
    full_text = processor.apply_chat_template(messages_full, tokenize=False)
    inputs    = processor(text=full_text, images=[image], return_tensors="pt")
    inputs    = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in inputs.items()
    }

    # Compute prompt length (needed to mask caption-only labels)
    messages_prompt = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt_text    = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    prompt_inputs  = processor(text=prompt_text, images=[image], return_tensors="pt")
    prompt_len     = prompt_inputs["input_ids"].shape[1]

    # Caption mask — True for caption tokens only
    caption_labels = inputs["input_ids"].clone()
    caption_labels[:, :prompt_len] = -100
    caption_labels[caption_labels == processor.tokenizer.pad_token_id] = -100
    cap_mask = (caption_labels[..., 1:] != -100)
    if cap_mask.sum() == 0:
        cap_mask = torch.ones_like(caption_labels[..., 1:], dtype=torch.bool)

    return inputs, prompt_len, cap_mask, caption_labels


def _forward_with_noise(model, inputs, caption_labels, sigma: float):
    """
    Run a forward pass with Gaussian noise injected into the vision encoder
    output (image embeddings), before they are passed to the language model.

    Uses a PyTorch forward hook registered on model.model.vision_model.
    The hook adds N(0, sigma^2) noise to the last hidden state.

    Returns per-token log-probabilities for caption tokens.
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    def _hook(module, input, output):
        """Intercept vision encoder output and add Gaussian noise."""
        # SmolVLM vision encoder returns a ModelOutput object
        # The last_hidden_state is what gets projected to the LLM space
        if hasattr(output, "last_hidden_state"):
            noise = torch.randn_like(output.last_hidden_state) * sigma
            # Return a new object with noisy hidden state
            # We modify in-place on a clone to avoid affecting the original
            output.last_hidden_state = output.last_hidden_state + noise
            return output
        # Fallback: if output is a plain tensor
        if isinstance(output, torch.Tensor):
            return output + torch.randn_like(output) * sigma
        return output

    hook_handle = model.model.vision_model.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            out = model(**inputs, labels=caption_labels)
        logits       = out.logits                             # [1, seq_len, vocab]
        shift_logits = logits[..., :-1, :]                   # [1, seq_len-1, vocab]
        shift_labels = inputs["input_ids"][..., 1:]          # [1, seq_len-1]
        token_probs  = F.softmax(shift_logits, dim=-1)
        actual_probs = token_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]
        log_probs    = torch.log(actual_probs + 1e-10)       # [1, seq_len-1]
    finally:
        hook_handle.remove()

    return log_probs  # [1, seq_len-1]


def _forward_original(model, inputs, caption_labels):
    """
    Standard forward pass — no noise.
    Returns per-token log-probabilities.
    """
    with torch.inference_mode():
        out = model(**inputs, labels=caption_labels)
    logits       = out.logits
    shift_logits = logits[..., :-1, :]
    shift_labels = inputs["input_ids"][..., 1:]
    token_probs  = F.softmax(shift_logits, dim=-1)
    actual_probs = token_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return torch.log(actual_probs + 1e-10)  # [1, seq_len-1]


def compute_pcmmia_features(model, processor, raw_image, text: str) -> dict:
    """
    Compute PC-MMIA features for a single image-caption pair.

    Algorithm (Zheng et al., 2025):
      1. Run original forward pass → log p(yi; original) for each token
      2. Run forward pass with +sigma noise on image embeddings → log p(yi; x+)
      3. Run forward pass with -sigma noise on image embeddings → log p(yi; x-)
      4. Compute per-token delta:
           δ(yi) = log p(yi; original) - 0.5*(log p(yi; x+) + log p(yi; x-))
      5. Softmax-normalize delta values → calibration weights wi
      6. Final score = sum(wi * log p(yi; original))

    High score → tokens are sensitive to image perturbation → likely memorized → member.
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    image  = _load_image(raw_image)

    inputs, prompt_len, cap_mask, caption_labels = _prepare_inputs(
        processor, image, text, device, dtype
    )

    # ── Three forward passes ───────────────────────────────────────────────────
    log_probs_orig  = _forward_original(model, inputs, caption_labels)       # [1, seq_len-1]
    log_probs_plus  = _forward_with_noise(model, inputs, caption_labels,  SIGMA)
    log_probs_minus = _forward_with_noise(model, inputs, caption_labels, -SIGMA)

    # ── Extract caption tokens only ────────────────────────────────────────────
    cap_mask_1d = cap_mask[0]  # [seq_len-1]

    lp_orig  = log_probs_orig[0][cap_mask_1d].float()   # [n_cap_tokens]
    lp_plus  = log_probs_plus[0][cap_mask_1d].float()
    lp_minus = log_probs_minus[0][cap_mask_1d].float()

    # ── Per-token delta (Eq. 3 in paper) ──────────────────────────────────────
    delta = lp_orig - 0.5 * (lp_plus + lp_minus)        # [n_cap_tokens]

    # ── Softmax-normalised weights (Eq. 4) ────────────────────────────────────
    weights = torch.softmax(delta, dim=0)                # [n_cap_tokens]

    # ── Final membership score (Eq. 5) ────────────────────────────────────────
    pcmmia_score = (weights * lp_orig).sum().item()

    # ── Summary statistics of delta ───────────────────────────────────────────
    n    = delta.shape[0]
    k10  = max(1, int(n * 0.10))
    k25  = max(1, int(n * 0.25))
    top_delta_sorted = delta.sort(descending=True).values

    return {
        "pcmmia_score":       pcmmia_score,
        "pcmmia_mean_delta":  delta.mean().item(),
        "pcmmia_max_delta":   delta.max().item(),
        "pcmmia_std_delta":   delta.std().item(),
        "pcmmia_top10_delta": top_delta_sorted[:k10].mean().item(),
        "pcmmia_top25_delta": top_delta_sorted[:k25].mean().item(),
    }


# ── Extraction pipeline with checkpointing ────────────────────────────────────

_EMPTY_ROW = {
    "pcmmia_score":       0.0,
    "pcmmia_mean_delta":  0.0,
    "pcmmia_max_delta":   0.0,
    "pcmmia_std_delta":   0.0,
    "pcmmia_top10_delta": 0.0,
    "pcmmia_top25_delta": 0.0,
}


def extract_pcmmia_for_split(
    model,
    processor,
    df: pd.DataFrame,
    save_path: str,
    has_labels: bool = True,
    checkpoint_every: int = 100,
):
    """
    Extract PC-MMIA features for an entire dataset split with incremental saves.

    Resumes automatically from an existing partial CSV — just re-run the same
    call after a disconnect and it will pick up where it left off.

    Args:
        model:            Fine-tuned SmolVLM model
        processor:        Corresponding processor
        df:               DataFrame with 'image', 'text', 'id' columns
        save_path:        Where to save the CSV (also used for resuming)
        has_labels:       Whether df has 'is_member' column
        checkpoint_every: Save to disk every N samples
    """
    # ── Resume from checkpoint if it exists ───────────────────────────────────
    start_idx = 0
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        start_idx = len(existing)
        print(f"Resuming from sample {start_idx} / {len(df)}")
    else:
        print(f"Starting fresh — {len(df)} samples to process")

    texts  = df["text"].tolist()
    images = df["image"].tolist()
    ids    = df["id"].tolist()
    labels = df["is_member"].tolist() if has_labels else [0] * len(df)

    buffer = []

    for i in tqdm(range(start_idx, len(df)), desc="PC-MMIA extraction"):
        try:
            feats = compute_pcmmia_features(model, processor, images[i], texts[i])
        except Exception as e:
            print(f"\nError on id={ids[i]}: {str(e)[:120]}")
            feats = dict(_EMPTY_ROW)

        feats["id"]    = ids[i]
        feats["label"] = labels[i]
        buffer.append(feats)

        # ── Checkpoint save ────────────────────────────────────────────────────
        if len(buffer) >= checkpoint_every:
            chunk_df = pd.DataFrame(buffer)
            mode   = "w" if (i - len(buffer) + 1 == start_idx and start_idx == 0) else "a"
            header = not os.path.exists(save_path) or mode == "w"
            chunk_df.to_csv(save_path, mode=mode, header=header, index=False)
            print(f"\nCheckpoint saved — {i + 1} / {len(df)} samples done")
            buffer = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save any remaining samples in buffer ──────────────────────────────────
    if buffer:
        chunk_df = pd.DataFrame(buffer)
        mode   = "w" if start_idx == 0 and not os.path.exists(save_path) else "a"
        header = not os.path.exists(save_path) or mode == "w"
        chunk_df.to_csv(save_path, mode=mode, header=header, index=False)
        print(f"\nFinal save — {len(df)} samples complete.")

    print(f"Done. CSV saved to {save_path}")
