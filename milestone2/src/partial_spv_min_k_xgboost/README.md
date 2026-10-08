This folder contains all of the code for the Milestone 2 model `milestone2/partial_spv_min_k_xgboost`. 

To run the full pipeline, type `python main.py` in your terminal after installing the necessary 
dependencies in `requirements.txt`. 

For the full walk-through of the model training and evaluation process used (in Colab), please see the `partial_spv_min_k_xgboost.ipynb` file.

This model builds on the `min_k_xgboost` baseline by adding probabilistic variation features 
inspired by the SPV-MIA method proposed by Fu et al. (NeurIPS 2024). For each text, slightly 
modified versions are generated using the T5 model for masking, and the change in loss between the original 
and perturbed texts is used as an additional feature for membership signaling. The SmolLM2-360M model serves as 
the target model and SmolLM2-135M as the reference model. Due to computational constraints, SPV features were 
computed on a subset of 3,000 training samples, with remaining samples set to 0 values.

The code is organised as follows:
- `config.py` — all global variables and hyperparameters
- `compute_perplexity_and_features.py` — base feature extraction (perplexity, loss, Min-K% etc.)
- `compute_spv.py` — SPV-MIA probabilistic variation feature extraction, adapted from Fu et al. (2024)
- `batches.py` — batch processing 
- `build_membership_classifier.py` — XGBoost, Logistic Regression and Random Forest classifiers
- `main.py` — full pipeline