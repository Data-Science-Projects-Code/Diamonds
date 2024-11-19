import pandas as pd
from data_prep import load_and_prepare_data
import os

def split_by_iqr(df, column='carat_weight'):
    """
    Splits a DataFrame into low and high subsets based on IQR cutoff.
    
    Parameters:
    df (DataFrame): Input DataFrame to split.
    column (str): The column to calculate IQR and split on.
    
    Returns:
    tuple: (low_df, high_df)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    
    # Splitting
    low_df = df[df[column] < upper_limit].copy()
    high_df = df[df[column] >= upper_limit].copy()
    
    # Finding new IQR for high_df
    if not high_df.empty:
        new_Q1 = high_df[column].quantile(0.25)
        new_Q3 = high_df[column].quantile(0.75)
        high_df = high_df[high_df[column] >= new_Q1]  # Ensuring continuity
    
    return low_df, high_df

if __name__ == "__main__":
    # Load data
    clear_diamonds, fancy_diamonds = load_and_prepare_data("../data/diamonds.parquet")
    
    # Split data
    clear_low, clear_high = split_by_iqr(clear_diamonds, column='log_carat_weight')
    fancy_low, fancy_high = split_by_iqr(fancy_diamonds, column='log_carat_weight')
    
    # Print shapes for verification
    print(f"Clear Low: {clear_low.shape}, Clear High: {clear_high.shape}")
    print(f"Fancy Low: {fancy_low.shape}, Fancy High: {fancy_high.shape}")
    
    # Save the splits to Parquet files
    output_dir = "../data/splits"
    os.makedirs(output_dir, exist_ok=True)
    
    clear_low.to_parquet(os.path.join(output_dir, "clear_low.parquet"))
    clear_high.to_parquet(os.path.join(output_dir, "clear_high.parquet"))
    fancy_low.to_parquet(os.path.join(output_dir, "fancy_low.parquet"))
    fancy_high.to_parquet(os.path.join(output_dir, "fancy_high.parquet"))

    print("Data splits have been saved to Parquet files.")

