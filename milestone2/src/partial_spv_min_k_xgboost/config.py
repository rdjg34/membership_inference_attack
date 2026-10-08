"""
Config: all global variables.
"""

# Models & dataset
MODEL_135M = "UBC-SLIME/colx_531_smollm2-135m"
MODEL_360M = "UBC-SLIME/colx_531_smollm2-360m"
TOKENIZER  = "UBC-SLIME/colx_531_smollm2-135m"
DATASET    = "UBC-SLIME/colx_531_group_project"

# HuggingFace Hub
HF_REPO_ID = "rdj-034/mia-medical-llm"  # updated from placeholder

# W&B
WANDB_PROJECT  = "mia-tcob"
WANDB_RUN_NAME = "smollm2-135m-360m-spv-mia"

# Pipeline
BATCH_SIZE      = 500
MAX_TOKEN_LENGTH = 512

# SPV-MIA parameters
SPV_NUM_PERTURBATIONS = 2
SPV_MASK_PCT          = 0.3
SPV_SPAN_LENGTH       = 2
SPV_MASK_MODEL        = "t5-base"
SPV_T5_MAX_LENGTH     = 300

# SPV subset sizes
SPV_N_TRAIN = 3000
SPV_N_VAL   = 500
SPV_N_TEST  = 500

# XGBoost parameters
XGB_N_ESTIMATORS    = 200
XGB_LEARNING_RATE   = 0.05
XGB_MAX_DEPTH       = 4
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE_BYTREE = 0.8