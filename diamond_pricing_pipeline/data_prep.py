import pandas as pd
import numpy as np

def load_and_prepare_data(file_path):
    """
    Reads a parquet file and splits it into clear and fancy diamond datasets.
    
    Parameters:
    file_path (str): Path to the parquet file.
    
    Returns:
    tuple: (all_clear_diamonds, all_fancy_diamonds)
    """
    df = pd.read_parquet(file_path)
    
    # Log transformations
    df['log_total_sales_price'] = np.log(df['total_sales_price'])
    df['log_carat_weight'] = np.log(df['carat_weight'])
    
    # Splitting datasets
    all_clear_diamonds = df[(df['fancy_color_dominant_color'] == 'unknown') & (df['color'] != 'unknown')].copy()
    all_fancy_diamonds = df[(df['fancy_color_dominant_color'] != 'unknown') & (df['color'] == 'unknown')].copy()
    
    return all_clear_diamonds, all_fancy_diamonds

if __name__ == "__main__":
    file_path = "../data/diamonds.parquet"
    clear_diamonds, fancy_diamonds = load_and_prepare_data(file_path)
    print(f"Clear diamonds shape: {clear_diamonds.shape}")
    print(f"Fancy diamonds shape: {fancy_diamonds.shape}")

