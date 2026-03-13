import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_perplexity_and_features(model, tokenizer, text, max_length=512):
    """
    Compute perplexity and other features that might indicate memorization.

    Returns:
        dict with perplexity, loss, confidence metrics, Min-K% features

    """
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
    }

    return result
