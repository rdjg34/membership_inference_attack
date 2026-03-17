import torch
import torch.nn.functional as F


def compute_perplexity_and_features(model, tokenizer, text, max_length=512,
                                    ref_model=None):
    """
    Compute perplexity and other features that might indicate memorization.

    Args:
        model:      Fine-tuned target model.
        tokenizer:  Shared tokenizer.
        text:       Input string.
        max_length: Max token length.
        ref_model:  Optional pre-trained reference model for HT-MIA features.
                    If provided, adds ht_mia_score_k10/20/30 to the result.

    Returns:
        dict with perplexity, loss, confidence metrics, Min-K% features,
        and (optionally) HT-MIA features.

    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

        # Lower loss, model has likely seen this pattern before
        loss = outputs.loss.item()

        # Low perplexity, more likely seen before
        perplexity = torch.exp(outputs.loss).item()

        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]
        token_probs = F.softmax(shift_logits, dim=-1)

        # Probability of the actual token at each position
        actual_token_probs = token_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Average confidence across all tokens
        mean_token_prob = actual_token_probs.mean().item()

        # Lowest / highest single-token confidence
        min_token_prob = actual_token_probs.min().item()
        max_token_prob = actual_token_probs.max().item()

        # Lower std, uniformly high confidence, more likely seen before
        std_token_prob = actual_token_probs.std().item()

        # Higher entropy, more uncertain, less likely seen before
        token_entropies = -(token_probs * torch.log(token_probs + 1e-10)).sum(dim=-1)
        mean_entropy = token_entropies.mean().item()

        # High top-k mass, confident and likely seen before
        top_k_probs, _ = torch.topk(token_probs, k=10, dim=-1)
        top_k_mass = top_k_probs.sum(dim=-1).mean().item()

        # Low ratio, few uncertain tokens, likely seen before
        low_conf_tokens = (actual_token_probs < 0.1).sum().item()
        low_conf_ratio = low_conf_tokens / actual_token_probs.shape[1]

        # Min-K% features
        # Average probability of the bottom K% of tokens.
        # Members have higher confidence even on their hardest tokens.
        sorted_probs = actual_token_probs.sort(dim=-1).values  # ascending
        n_tokens = sorted_probs.shape[1]

        k10 = max(1, int(n_tokens * 0.10))
        min_k10_prob = sorted_probs[:, :k10].mean().item()

        k20 = max(1, int(n_tokens * 0.20))
        min_k20_prob = sorted_probs[:, :k20].mean().item()

        k30 = max(1, int(n_tokens * 0.30))
        min_k30_prob = sorted_probs[:, :k30].mean().item()

    # HT-MIA (Hard Token Membership Inference Attack)
    # This approach is from
    # Raw perplexity is a noisy membership signal because some text is simply
    # easier or harder to predict regardless of whether the model was trained on
    # it.  HT-MIA corrects for this by comparing the fine-tuned target model
    # against a reference model (the same architecture but *not* fine-tuned).
    #
    # If a sequence was in the training set, the fine-tuned target
    # should assign higher token probabilities than the untuned reference —
    # especially on the tokens that are hardest to predict (the bottom-K%).
    # Non-members won't show this systematic improvement over the reference.
    #
    # Score = fraction of the bottom-K% hardest tokens (by target probability)
    #         on which the target outperforms the reference.
    # Higher score → more likely a member.
    #
    # Implemented after reading the following article:
    # Reference: Md Tasnim Jawad, Mingyan Xiao, and Yanzhao
    # Wu. 2026. What Hard Tokens Reveal: Exploiting
    # Low-Confidence Tokens for Membership Inference
    # Attacks against Large Language Models. arXiv
    # preprint arXiv:2601.20885.

    ht_mia_scores = {}
    if ref_model is not None:
        with torch.no_grad():
            # Score the same input with the reference (pre-trained, not fine-tuned) model
            ref_outputs = ref_model(**inputs, labels=inputs["input_ids"])
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[..., :-1, :]
            ref_token_probs = F.softmax(ref_shift_logits, dim=-1)
            # Per-token probability under the reference model
            ref_actual_probs = ref_token_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

        # Positive improvement means the fine-tuned target is more confident
        # than the reference on that token — a potential memorization signal
        improvement = actual_token_probs - ref_actual_probs  # positive = target wins

        # Focus on the hardest tokens: sort ascending so index 0 = lowest target prob
        sort_idx = actual_token_probs.argsort(dim=-1)  # ascending = hardest first
        for k_pct, k_n in [("k10", k10), ("k20", k20), ("k30", k30)]:
            hard_idx = sort_idx[:, :k_n]          # indices of the K hardest tokens
            hard_improvement = improvement.gather(1, hard_idx)
            # Fraction of hard tokens where target beats reference (range 0–1)
            ht_mia_scores[f"ht_mia_score_{k_pct}"] = (
                (hard_improvement > 0).float().mean().item()
            )

    result = {
        "perplexity":      perplexity,
        "loss":            loss,
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
        "num_tokens":      n_tokens,
        **ht_mia_scores,
    }

    return result
