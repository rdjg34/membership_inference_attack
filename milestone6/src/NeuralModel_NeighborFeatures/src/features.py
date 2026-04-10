import io
import zlib
import torch
import torch.nn.functional as F
from PIL import Image
from config import RENYI_ALPHAS, RENYI_K_PERCENTS


def _load_image(raw_image) -> Image.Image:
    if isinstance(raw_image, dict) and "bytes" in raw_image:
        img = Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
    else:
        img = raw_image.convert("RGB")
    return img


def _prepare_inputs(processor, image: Image.Image, text: str, device: torch.device):
    """
    Split text into user prompt / assistant caption, apply chat template,
    and return (inputs, prompt_len).
    """
    parts = text.split("\n", 1)
    user_text      = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    messages_full = [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text",  "text": assistant_text}]},
    ]
    full_text = processor.apply_chat_template(messages_full, tokenize=False)
    inputs = processor(text=full_text, images=[image], return_tensors="pt")

    inputs = {
        k: v.to(device=device, dtype=torch.bfloat16)
           if v.is_floating_point()
           else v.to(device=device)
        for k, v in inputs.items()
    }

    messages_prompt = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    return inputs, prompt_len


def _renyi_entropy(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute Renyi entropy of order alpha for each token position.

    Args:
        probs: (num_tokens, vocab_size) probability distributions
        alpha: Renyi order. Must not be 1.0 (use Shannon entropy instead).

    Returns:
        (num_tokens,) Renyi entropy at each position
    """
    # Clamp to avoid log(0)
    probs = probs.clamp(min=1e-10)
    # Sum alpha at each token position:
    sum_p_alpha = (probs ** alpha).sum(dim=-1)
    return (1.0 / (1.0 - alpha)) * torch.log(sum_p_alpha)


def _compute_loss_features(model, processor, inputs, prompt_len):
    """
    Core loss and token-probability features from a single model.
    Returns a dict of features.
    """
    device = next(model.parameters()).device

    # Caption-only labels (mask prompt + padding)
    caption_labels = inputs["input_ids"].clone()
    caption_labels[:, :prompt_len] = -100
    caption_labels[caption_labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        out = model(**inputs, labels=caption_labels)
        caption_loss = out.loss.item()
        caption_perplexity = torch.exp(out.loss.detach()).item()

        # Per-token probabilities (caption tokens only)
        logits = out.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]

        token_probs = F.softmax(shift_logits, dim=-1)
        actual_token_probs = token_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        cap_mask = (caption_labels[..., 1:] != -100)
        if cap_mask.sum() == 0:
            cap_mask = torch.ones_like(caption_labels[..., 1:], dtype=torch.bool)

        cap_probs = actual_token_probs[cap_mask].to(device=device, dtype=torch.float32)
        caption_length = int(cap_probs.numel())

        # Basic probability stats
        mean_token_prob = cap_probs.mean().item()
        std_token_prob  = cap_probs.std().item()
        min_token_prob  = cap_probs.min().item()

        # Min-K% (most uncertain tokens — key signal for MIA)
        sorted_probs = cap_probs.sort().values
        n = sorted_probs.shape[0]
        k10 = max(1, int(n * 0.10))
        k20 = max(1, int(n * 0.20))
        min_k10_prob = sorted_probs[:k10].mean().item()
        min_k20_prob = sorted_probs[:k20].mean().item()

        # Entropy
        cap_full_dist = token_probs[0][cap_mask[0]].to(device=device, dtype=torch.float32)
        token_entropies = -(cap_full_dist * torch.log(cap_full_dist + 1e-10)).sum(dim=-1)
        mean_entropy = token_entropies.mean().item()

        # Token rank features
        # actual_ranks[i] = number of vocab tokens with strictly higher prob than the actual token + 1
        # rank 1 = model's single most likely prediction at that position
        actual_ranks = (cap_full_dist > cap_probs.unsqueeze(-1)).sum(dim=-1) + 1
        median_token_rank = float(actual_ranks.float().median().item())
        frac_rank1 = float((actual_ranks == 1).float().mean().item())
        frac_rank5 = float((actual_ranks <= 5).float().mean().item())

        # First-K mean probability: confidence on the opening tokens of the caption
        K = min(5, cap_probs.shape[0])
        first_k_mean_prob = float(cap_probs[:K].mean().item())

        # Prefix / suffix probability ratio: members tend to have more uniform confidence
        mid = cap_probs.shape[0] // 2
        if mid > 0:
            prefix_suffix_ratio = float(
                cap_probs[:mid].mean().item() / max(cap_probs[mid:].mean().item(), 1e-10)
            )
        else:
            prefix_suffix_ratio = 1.0

        # Renyi entropy features (Zhu et al. 2025, Li et al. 2024)
        # (Walked through this implementation with Claude to understand the logic)
        renyi_features = {}
        for alpha in RENYI_ALPHAS:
            alpha_key = str(alpha).replace(".", "")  # eg, 0.5 becomes "05"
            token_renyi = _renyi_entropy(cap_full_dist, alpha)

            # ModRenyi: mean Renyi entropy across all caption tokens
            renyi_features[f"mod_renyi_a{alpha_key}"] = token_renyi.mean().item()

            # MaxRenyi-K%: sort descending, take top-K% (highest entropy tokens)
            sorted_renyi = token_renyi.sort(descending=True).values
            n_renyi = sorted_renyi.shape[0]
            for k_pct in RENYI_K_PERCENTS:
                k_count = max(1, int(n_renyi * k_pct))
                k_key = str(int(k_pct * 100))
                renyi_features[f"max_renyi_k{k_key}_a{alpha_key}"] = (
                    sorted_renyi[:k_count].mean().item()
                )

        # Temperature sensitivity (Hu et al. 2025)
        # Members have more peaked distributions → loss drops more sharply at low T
        # Got help from Claude to create this implementation.
        temp_losses = {}
        for T in [0.25, 0.5, 2.0, 4.0]:
            t_probs = F.softmax(shift_logits / T, dim=-1)
            t_cap_probs = t_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            t_cap_probs = t_cap_probs[cap_mask].to(device=device, dtype=torch.float32)
            temp_losses[T] = -torch.log(t_cap_probs + 1e-10).mean().item()
        loss_at_T025 = temp_losses[0.25]
        loss_at_T05  = temp_losses[0.5]
        loss_at_T20  = temp_losses[2.0]
        loss_at_T40  = temp_losses[4.0]
        # Sensitivity = low-T loss minus high-T loss (more negative → more peaked → member)
        temp_sensitivity     = loss_at_T05  - loss_at_T20
        temp_sensitivity_ext = loss_at_T025 - loss_at_T40

    result = {
        "caption_loss": caption_loss,
        "caption_perplexity": caption_perplexity,
        "caption_length": caption_length,
        "mean_token_prob": mean_token_prob,
        "std_token_prob": std_token_prob,
        "min_token_prob": min_token_prob,
        "min_k10_prob": min_k10_prob,
        "min_k20_prob": min_k20_prob,
        "mean_entropy": mean_entropy,
        "loss_at_T025": loss_at_T025,
        "loss_at_T05": loss_at_T05,
        "loss_at_T20": loss_at_T20,
        "loss_at_T40": loss_at_T40,
        "temp_sensitivity": temp_sensitivity,
        "temp_sensitivity_ext": temp_sensitivity_ext,
        "first_k_mean_prob": first_k_mean_prob,
        "prefix_suffix_ratio": prefix_suffix_ratio,
        "median_token_rank": median_token_rank,
        "frac_rank1": frac_rank1,
        "frac_rank5": frac_rank5,
    }
    result.update(renyi_features)
    return result


def compute_vlm_features(model, processor, raw_image, text: str):
    """
    Compute features from the fine-tuned (target) model.
    """
    device = next(model.parameters()).device
    image  = _load_image(raw_image)
    inputs, prompt_len = _prepare_inputs(processor, image, text, device)
    feats = _compute_loss_features(model, processor, inputs, prompt_len)

    # Zlib compression ratio: lower model loss relative to text compressibility = member signal
    caption_text = text.split("\n", 1)[1] if "\n" in text else text
    zlib_bits = len(zlib.compress(caption_text.encode("utf-8"))) * 8
    feats["zlib_entropy"] = zlib_bits / max(len(caption_text), 1)
    feats["zlib_ratio"]   = feats["caption_loss"] / max(feats["zlib_entropy"], 1e-10)

    return feats


def compute_neighbor_losses_batched(
    model, processor, raw_image, user_text: str, neighbor_captions: list[str]
) -> list[float]:
    """
    Compute caption loss for each neighbor caption in one batched forward pass.

    All neighbors share the same image and user prompt, so prompt_len is
    identical across the batch — right-side padding does not affect the
    caption token mask.

    Returns a list of per-caption losses in the same order as neighbor_captions.
    """
    device = next(model.parameters()).device
    image  = _load_image(raw_image)

    # Prompt length is the same for every neighbor (same image + user_text).
    messages_prompt = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    prompt_len = processor(
        text=prompt_text, images=[image], return_tensors="pt"
    )["input_ids"].shape[1]

    # Build one input per neighbor caption.
    full_texts = []
    for cap in neighbor_captions:
        messages_full = [
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text",  "text": cap}]},
        ]
        full_texts.append(processor.apply_chat_template(messages_full, tokenize=False))

    inputs = processor(
        text=full_texts,
        images=[[image] for _ in neighbor_captions],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(device=device, dtype=torch.bfloat16)
           if v.is_floating_point()
           else v.to(device=device)
        for k, v in inputs.items()
    }

    losses = []
    with torch.no_grad():
        logits = model(**inputs).logits  # (N, L, vocab)

        for i in range(len(neighbor_captions)):
            input_ids_i = inputs["input_ids"][i]
            attn_mask_i = inputs["attention_mask"][i]

            caption_labels = input_ids_i.clone()
            caption_labels[attn_mask_i == 0] = -100   # mask padding (left or right)

            # Find where real content starts — handles both left- and right-padding.
            real_token_positions = (attn_mask_i == 1).nonzero(as_tuple=True)[0]
            first_real = int(real_token_positions[0].item())
            caption_labels[:first_real + prompt_len] = -100  # mask padding + prompt

            pad_id = processor.tokenizer.pad_token_id
            if pad_id is not None:
                caption_labels[caption_labels == pad_id] = -100

            shift_logits = logits[i, :-1, :].float()
            shift_labels = caption_labels[1:]

            cap_mask = shift_labels != -100
            if cap_mask.sum() == 0:
                losses.append(float("inf"))
                continue

            losses.append(
                F.cross_entropy(shift_logits[cap_mask], shift_labels[cap_mask]).item()
            )

    return losses


def compute_reference_features(base_model, base_processor, raw_image, text: str):
    """
    Compute the same core features from the BASE (reference) model.
    """
    device = next(base_model.parameters()).device
    image  = _load_image(raw_image)
    inputs, prompt_len = _prepare_inputs(base_processor, image, text, device)
    feats = _compute_loss_features(base_model, base_processor, inputs, prompt_len)

    # Prefix with "ref_" to distinguish from target model
    return {f"ref_{k}": v for k, v in feats.items()}


def compute_comparative_features(target_feats: dict, ref_feats: dict) -> dict:
    """
    Compute difference / ratio features between fine-tuned and base model.

    This is the primary signal for membership inference:
    - Members → fine-tuned model has LOWER loss than base model
    - Non-members → both models have SIMILAR loss
    """
    t_loss = target_feats["caption_loss"]
    r_loss = ref_feats["ref_caption_loss"]

    result = {
        # Loss difference (members → negative)
        "caption_loss_diff": t_loss - r_loss,

        # Loss ratio (members → < 1.0)
        "loss_ratio": t_loss / max(r_loss, 1e-10),

        # Perplexity ratio
        "perplexity_ratio": target_feats["caption_perplexity"] / max(
            ref_feats["ref_caption_perplexity"], 1e-10
        ),

        # Token probability difference (members → positive)
        "mean_prob_diff": target_feats["mean_token_prob"] - ref_feats["ref_mean_token_prob"],

        # Min-K% difference
        "min_k10_prob_diff": target_feats["min_k10_prob"] - ref_feats["ref_min_k10_prob"],
        "min_k20_prob_diff": target_feats["min_k20_prob"] - ref_feats["ref_min_k20_prob"],

        # Entropy difference (members → negative = lower entropy)
        "mean_entropy_diff": target_feats["mean_entropy"] - ref_feats["ref_mean_entropy"],

        # Temperature sensitivity diff (members → target more sensitive than base)
        "temp_sensitivity_diff":     target_feats["temp_sensitivity"]     - ref_feats["ref_temp_sensitivity"],
        "temp_sensitivity_ext_diff": target_feats["temp_sensitivity_ext"] - ref_feats["ref_temp_sensitivity_ext"],

        # Token-rank diffs (members → higher frac_rank1/5, lower median_token_rank)
        "first_k_mean_prob_diff":   target_feats["first_k_mean_prob"]   - ref_feats["ref_first_k_mean_prob"],
        "prefix_suffix_ratio_diff": target_feats["prefix_suffix_ratio"] - ref_feats["ref_prefix_suffix_ratio"],
        "median_token_rank_diff":   target_feats["median_token_rank"]   - ref_feats["ref_median_token_rank"],
        "frac_rank1_diff":          target_feats["frac_rank1"]          - ref_feats["ref_frac_rank1"],
        "frac_rank5_diff":          target_feats["frac_rank5"]          - ref_feats["ref_frac_rank5"],
    }

    # Renyi comparative features:
    # Members should show lower Renyi entropy on the fine-tuned model (negative diff)
    for alpha in RENYI_ALPHAS:
        alpha_key = str(alpha).replace(".", "")
        mod_key = f"mod_renyi_a{alpha_key}"
        if mod_key in target_feats and f"ref_{mod_key}" in ref_feats:
            result[f"{mod_key}_diff"] = target_feats[mod_key] - ref_feats[f"ref_{mod_key}"]

        for k_pct in RENYI_K_PERCENTS:
            k_key = str(int(k_pct * 100))
            mr_key = f"max_renyi_k{k_key}_a{alpha_key}"
            if mr_key in target_feats and f"ref_{mr_key}" in ref_feats:
                result[f"{mr_key}_diff"] = target_feats[mr_key] - ref_feats[f"ref_{mr_key}"]

    return result
