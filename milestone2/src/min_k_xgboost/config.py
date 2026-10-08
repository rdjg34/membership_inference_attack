"""
Config: all global variables.
"""

# Models & dataset
MODEL_135M = "UBC-SLIME/colx_531_smollm2-135m"
MODEL_360M = "UBC-SLIME/colx_531_smollm2-360m"
TOKENIZER = "UBC-SLIME/colx_531_smollm2-135m"
DATASET = "UBC-SLIME/colx_531_group_project"

# HuggingFace Hub
HF_REPO_ID = "username/mia-medical-llm"  # replace with your username

# W&B
WANDB_PROJECT = "mia-tcob"
WANDB_RUN_NAME = "smollm2-135m-360m-recall-mink"
