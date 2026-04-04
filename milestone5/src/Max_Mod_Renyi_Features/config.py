"""
Config: all global variables.
"""

# Fine-tuned VLM target model
MODEL_ID = "UBC-SLIME/colx_585_vlm"

# Base (pre-trained) model for reference comparison
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Base"

# Dataset
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

# Renyi entropy configuration (can be adjusted if needed)
RENYI_ALPHAS = [0.5, 2.0]
RENYI_K_PERCENTS = [0.10, 0.20]