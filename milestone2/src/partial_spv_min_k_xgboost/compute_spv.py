# SPV-MIA functions - adapted from code provided by authors of
# Fu et al (NeurIPS 2024) paper at:
# https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA/tree/main
# specifically, attack_model.py, with some support from Claude AI

import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# these will be set by main.py before calling the below functions
model_135m = None
model_360m = None
tokenizer  = None
device     = None
t5_model   = None
t5_tokenizer = None

# ── Step 1 ────────────────────────────
def mask_text(text, mask_pct=0.3, span_length=2):
    """Replace ~30% of word spans with <extra_id_0>, <extra_id_1> etc."""
    words = text.split()

    if len(words) < 4:
        return text, 0

    n_spans = max(1, int(mask_pct * len(words) / (span_length + 2)))

    mask_string = "<<<mask>>>"
    n_masked = 0
    attempts = 0

    while n_masked < n_spans and attempts < 100:
        start = np.random.randint(0, max(1, len(words) - span_length))
        end = start + span_length
        search_start = max(0, start - 1)
        search_end = min(len(words), end + 1)
        if mask_string not in words[search_start:search_end]:
            words[start:end] = [mask_string]
            n_masked += 1
        attempts += 1

    num_masks = 0
    for idx, word in enumerate(words):
        if word == mask_string:
            words[idx] = f"<extra_id_{num_masks}>"
            num_masks += 1

    masked_text = " ".join(words)
    return masked_text, num_masks


# ── Step 2 ───────────────────────
def fill_masks_with_t5(masked_text, num_masks, n_perturbations=2):
    """Ask T5 to fill in the masked spans and return n filled texts."""

    stop_id = t5_tokenizer.encode(f"<extra_id_{num_masks}>")[0]

    # tokenize on CPU — no .to(device) since T5 is on CPU
    inputs = t5_tokenizer(
        masked_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # generate all perturbations in one T5 call
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_p=1.0,
            num_return_sequences=n_perturbations,
            eos_token_id=stop_id
        )

    # decode all outputs and return as a list
    filled_texts = [
        t5_tokenizer.decode(output, skip_special_tokens=False)
        .replace("<pad>", "").replace("</s>", "").strip()
        for output in outputs
    ]
    return filled_texts


# ── Step 3 ────────────────────────────
def reconstruct_text(masked_text, t5_output, num_masks):
    """Replace mask tags in the original text with T5's suggestions."""
    pattern = re.compile(r"<extra_id_\d+>")

    suggested_fills = [w.strip() for w in pattern.split(t5_output)[1:-1]]
    words = masked_text.split(" ")

    if len(suggested_fills) >= num_masks:
        for fill_idx in range(num_masks):
            tag = f"<extra_id_{fill_idx}>"
            if tag in words:
                words[words.index(tag)] = suggested_fills[fill_idx]
        return " ".join(words)
    else:
        print(f"Reconstruction failed: T5 gave {len(suggested_fills)} fills but need {num_masks}")
        return None


# ── Main perturbation function ────────────────
def generate_perturbations(text, n_perturbations=2):
    """
    Generate n slightly modified versions of the text.
    Each version has ~30% of its words replaced by T5.
    """
    try:
        # Step 1 — mask part of the text once
        masked_text, num_masks = mask_text(text)

        if num_masks == 0:
            return [text] * n_perturbations

        # Step 2 — ask T5 to fill in the blanks, getting n outputs in one call
        t5_outputs = fill_masks_with_t5(masked_text, num_masks, n_perturbations)

        # Step 3 — reconstruct each perturbed text from T5's outputs
        perturbed_texts = []
        for t5_output in t5_outputs:
            perturbed = reconstruct_text(masked_text, t5_output, num_masks)
            if perturbed and perturbed.strip():
                perturbed_texts.append(perturbed)
            else:
                perturbed_texts.append(text)

        return perturbed_texts

    except Exception as e:
        print(f"Perturbation failed: {str(e)[:80]}")
        return [text] * n_perturbations


# ── SPV feature computation ──────────────────────
def compute_spv_features(text, original_loss_360m=None,
                         original_loss_135m=None,
                         n_perturbations=2, max_length=512):
    """
    Compute SPV-MIA features for a single text.
    Accepts pre-computed original losses to avoid redundant forward passes.
    Uses 360M as target model and 135M as reference model.
    Returns spv_target, spv_reference, spv_calibrated.
    """

    def get_loss(model, text):
        """Get the model's loss on a single text."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():  # Option 2 — faster than no_grad
            outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

    try:
        # Option 1 — use pre-computed losses if available
        # saves 2 forward passes per text
        if original_loss_360m is None:
            original_loss_360m = get_loss(model_360m, text)
        if original_loss_135m is None:
            original_loss_135m = get_loss(model_135m, text)

        # generate perturbed versions of the text
        perturbed_texts = generate_perturbations(text, n_perturbations)

        # get loss on each perturbed version from both models
        perturbed_losses_360m = [get_loss(model_360m, p) for p in perturbed_texts]
        perturbed_losses_135m = [get_loss(model_135m, p) for p in perturbed_texts]

        # compute the average perturbed loss for each model
        mean_perturbed_360m = np.mean(perturbed_losses_360m)
        mean_perturbed_135m = np.mean(perturbed_losses_135m)

        # SPV = how much does loss increase when we perturb the text?
        # large SPV = steep probability peak = likely memorized = likely member
        spv_target    = mean_perturbed_360m - original_loss_360m
        spv_reference = mean_perturbed_135m - original_loss_135m

        # calibrated signal = remove natural text complexity
        # leaving only the memorization signal
        spv_calibrated = spv_target - spv_reference

        return {
            "spv_target":     spv_target,
            "spv_reference":  spv_reference,
            "spv_calibrated": spv_calibrated,
        }

    except Exception as e:
        print(f"SPV computation failed: {str(e)[:100]}")
        return {
            "spv_target":     0.0,
            "spv_reference":  0.0,
            "spv_calibrated": 0.0,
        }



def compute_spv_for_dataset(texts, labels, df_135m=None,
                             df_360m=None, batch_size=500,
                             save_dir=None):
    """
    Compute SPV-MIA features for all texts.
    Accepts pre-computed feature dataframes to reuse existing loss values.
    Saves each batch to Drive if save_dir is provided.
    Returns a DataFrame with spv_target, spv_reference, spv_calibrated, and label columns.
    """
    all_results = []
    print(f"Computing SPV features for {len(texts)} samples...")

    for i in range(0, len(texts), batch_size):
        end_idx       = min(i + batch_size, len(texts))
        batch_texts   = texts[i:end_idx]
        batch_labels  = labels[i:end_idx]
        batch_num     = i//batch_size + 1
        total_batches = (len(texts)-1)//batch_size + 1

        print(f"\nSPV Batch {batch_num}/{total_batches} ({end_idx - i} samples)")

        batch_results = []
        for j, (text, label) in enumerate(tqdm(zip(batch_texts, batch_labels),
                                               total=len(batch_texts),
                                               desc="Computing SPV features")):
            # Option 1 — reuse pre-computed losses if available
            global_idx = i + j
            loss_360m = df_360m['loss'].iloc[global_idx] if df_360m is not None else None
            loss_135m = df_135m['loss'].iloc[global_idx] if df_135m is not None else None

            spv_features = compute_spv_features(
                text,
                original_loss_360m=loss_360m,
                original_loss_135m=loss_135m
            )
            spv_features["label"] = label
            batch_results.append(spv_features)

        batch_df = pd.DataFrame(batch_results)
        all_results.append(batch_df)

        # save each batch to Drive immediately after completion
        if save_dir is not None:
            batch_path = f"{save_dir}spv_batch_{batch_num}.csv"
            batch_df.to_csv(batch_path, index=False)
            print(f"Batch {batch_num} saved to Drive: {batch_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_results = pd.concat(all_results, ignore_index=True)
    print(f"\nSPV feature computation complete: {len(final_results)} samples")
    return final_results