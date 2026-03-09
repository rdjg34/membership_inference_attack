from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def main():
    print("Loading dataset...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project")
    
    print("Dataset structure:")
    print(dataset)
    
    # Analyze splits
    stats = []
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        num_samples = len(df)
        
        # Assuming 'text' column exists based on typical NLP tasks
        # If not, we'll adjust after seeing the schema
        text_col = 'text' if 'text' in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
        
        if text_col:
            lengths = df[text_col].apply(lambda x: len(str(x)))
            stats.append({
                "Split": split,
                "Number of samples": num_samples,
                "Mean Length": lengths.mean(),
                "Max length": lengths.max(),
                "Min length": lengths.min()
            })
        else:
            stats.append({
                "Split": split,
                "Number of samples": num_samples,
                "Mean Length": "N/A",
                "Max length": "N/A",
                "Min length": "N/A"
            })
            
    stats_df = pd.DataFrame(stats)
    print("\nDescriptive Statistics:")
    print(stats_df)
    
    # Load models (optional for inspection, but required by lab)
    # print("\nLoading models (this might take a while)...")
    # lm1 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-135m")
    # lm2 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-360m")
    
    # Save a few samples for inspection.md
    print("\nSaving samples to milestone1/data_inspection.md...")
    with open("milestone1/data_inspection.md", "w") as f:
        f.write("# Data Inspection Samples\n\n")
        for split in dataset.keys():
            f.write(f"## Split: {split}\n\n")
            sample_df = dataset[split].to_pandas().head(3)
            f.write(sample_df.to_markdown(index=False))
            f.write("\n\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
