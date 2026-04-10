# Neural and Neighbors — MIA Pipeline

Membership inference attack pipeline combining teammate's three feature families
with VLM-based neighbor comparison features and an MLP stacking meta-learner.

This file was generateed by Claude


## Directory structure

```
NeuralModel_NeighborFeatures
  
    config.py               Global constants (model IDs, dataset ID, Renyi config)
    models.py               Loads fine-tuned and base SmolVLM models from HuggingFace
    features.py             Core VLM feature computation (entropy, Min-K%, Renyi, etc.)
    batches.py              Batched feature extraction + neighbor comparison
    extract_features.py     GPU script — generates *_neighbor_only.csv 
    create_merged_csvs.py   Merges all 3 family CSVs with neighbor CSVs → *_merged.csv
    lightweight_pipeline.py Teammate's RoutedBlend pipeline (max_features=100)
    mlp_classifier.py       MLP stacking meta-learner (LR + XGBoost → MLP)
    pipeline_with_neural.py Full CPU pipeline — reads CSVs, trains, writes submission
    run_pipeline.ipynb      Notebook to run the pipeline locally (no GPU needed)
    train_neighbor_only.csv Pre-extracted neighbor features 
    val_neighbor_only.csv
    test_neighbor_only.csv
    metrics_viewer.ipynb   Displays models' AUC and TPR@FPR=0.1 results
    
    
    ##Generated outputs
    
    submission.csv
    val_predictions.csv
    metrics.json

```

## How to run



### Run the pipeline (CPU, no GPU needed)



```bash
python pipeline_with_neural.py
```

Or open `run_pipeline.ipynb` and run all cells.



### Optional: inspect merged features

```bash
python create_merged_csvs.py
```

Saves `train/val/test_merged.csv` in `src/` — all three family CSVs joined with
neighbor features, for inspection. The pipeline does not read from these files;
it merges in memory.

### Optional: regenerate neighbor CSVs (requires GPU)

The `*_neighbor_only.csv` files are already provided. To regenerate from scratch:

```bash
python extract_features.py --n-neighbors 15
```

This loads the fine-tuned and base SmolVLM models from HuggingFace, runs inference
on the full dataset, and overwrites the existing neighbor CSVs.

## Data

The pipeline reads teammate's feature CSVs from:

```
../../milestone6/data/extracted_features/
  1_temperature/  train.csv  val.csv  test.csv
  2_renyi/        train.csv  val.csv  test.csv
  3_img_pert/     train.csv  val.csv  test.csv
```

To use a different data root:

```bash
python pipeline_with_neural.py --data-root /path/to/extracted_features
```

## Pipeline overview

```
Teammate's CSVs          Neighbor CSVs
(1_temperature,    +     (*_neighbor_only.csv)
 2_renyi,
 3_img_pert)
       │                        │
       └──────── merge ─────────┘
                   │
            add_id_features
            (target encoding)
                   │
          ┌────────┴────────┐
          │                 │
     RoutedBlend       MLP Stacking
     (LR + XGB +    (OOF LR + XGB
      per-source)    → MLP meta)
          │                 │
          └────── pick best ─┘
                   │
              submission.csv
```

## Neighbor features

The `*_neighbor_only.csv` files contain features computed by comparing each
sample's caption loss against 15 randomly sampled captions from the training set
on the same image. Members are expected to have lower loss than random captions.

| Feature | Description |
|---|---|
| `loss_delta_vs_neighbors` | Actual loss minus mean neighbor loss (members → negative) |
| `loss_ratio_vs_neighbors` | Actual loss / mean neighbor loss (members → < 1) |
| `loss_percentile_vs_neighbors` | Fraction of neighbors with lower loss (members → low) |
| `loss_std_vs_neighbors` | Std dev of neighbor losses (captures distribution spread) |

With n_neighbors=15, `loss_percentile_vs_neighbors` has 16 possible values
(vs 8 with n_neighbors=7), giving finer-grained signal.
