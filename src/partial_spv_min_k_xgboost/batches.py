import torch
import pandas as pd
from tqdm import tqdm

from compute_perplexity_and_features import compute_perplexity_and_features


def analyze_sample_batch(model, tokenizer, texts, labels, model_name="model"):
    """
    Analyze a batch of texts and return features for membership inference.

    """
    results = []

    for text, label in tqdm(zip(texts, labels), total=len(texts),
                            desc=f"Analyzing with {model_name}"):
        try:
            features = compute_perplexity_and_features(
                model, tokenizer, text
            )
            features["label"] = label
            features["text_length"] = len(text)
            results.append(features)
        except Exception as e:
            print(f"Error processing text (len={len(text)}): {str(e)[:100]}")
            results.append({
                "perplexity": float("inf"),
                "loss": float("inf"),
                "mean_token_prob": 0,
                "min_k10_prob": 0,
                "min_k20_prob": 0,
                "min_k30_prob": 0,
                "label": label,
                "text_length": len(text),
            })

    return pd.DataFrame(results)


def process_dataset_in_batches(model, tokenizer, texts, labels,
                               model_name="model", batch_size=1000):
    """
    Process large dataset in batches for memory efficiency.
    """
    all_results = []
    print(f"Processing {len(texts)} samples in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        end_idx     = min(i + batch_size, len(texts))
        batch_texts  = texts[i:end_idx]
        batch_labels = labels[i:end_idx]

        print(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: "
              f"{end_idx - i} samples")

        batch_results = analyze_sample_batch(
            model, tokenizer, batch_texts, batch_labels,
            model_name
        )
        all_results.append(batch_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_results = pd.concat(all_results, ignore_index=True)
    print(f"Completed processing {len(final_results)} samples with {model_name}")
    return final_results
