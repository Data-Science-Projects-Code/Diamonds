import pandas as pd
import pytest
import os
from split_data import split_by_iqr
from data_prep import load_and_prepare_data

@pytest.fixture
def mock_data():
    """
    Creates a mock dataset for testing.
    """
    data = {
        'total_sales_price': [1000, 2000, 3000, 4000, 5000],
        'carat_weight': [0.5, 1.0, 1.5, 2.0, 2.5],
        'fancy_color_dominant_color': ['unknown', 'unknown', 'blue', 'red', 'blue'],
        'color': ['D', 'E', 'unknown', 'unknown', 'unknown']
    }
    return pd.DataFrame(data)

def test_split_by_iqr(mock_data):
    """
    Tests the IQR-based splitting logic.
    """
    # Clear diamonds
    clear_diamonds = mock_data[(mock_data['fancy_color_dominant_color'] == 'unknown') & 
                               (mock_data['color'] != 'unknown')].copy()
    clear_low, clear_high = split_by_iqr(clear_diamonds, column='carat_weight')
    
    # Fancy diamonds
    fancy_diamonds = mock_data[(mock_data['fancy_color_dominant_color'] != 'unknown') & 
                               (mock_data['color'] == 'unknown')].copy()
    fancy_low, fancy_high = split_by_iqr(fancy_diamonds, column='carat_weight')
    
    # Assert splits for clear diamonds
    assert clear_low.shape[0] == 2  # Two rows in "low" split
    assert clear_high.shape[0] == 0  # No rows in "high" split
    
    # Assert splits for fancy diamonds
    assert fancy_low.shape[0] == 2  # Two rows in "low" split
    assert fancy_high.shape[0] == 1  # One row in "high" split

def test_save_parquet_files(tmpdir, mock_data):
    """
    Tests that Parquet files are saved correctly and match the expected data splits.
    """
    output_dir = tmpdir.mkdir("splits")
    clear_diamonds = mock_data[(mock_data['fancy_color_dominant_color'] == 'unknown') & 
                               (mock_data['color'] != 'unknown')].copy()
    fancy_diamonds = mock_data[(mock_data['fancy_color_dominant_color'] != 'unknown') & 
                               (mock_data['color'] == 'unknown')].copy()
    
    # Split data
    clear_low, clear_high = split_by_iqr(clear_diamonds, column='carat_weight')
    fancy_low, fancy_high = split_by_iqr(fancy_diamonds, column='carat_weight')
    
    # Save splits
    clear_low.to_parquet(os.path.join(output_dir, "clear_low.parquet"))
    clear_high.to_parquet(os.path.join(output_dir, "clear_high.parquet"))
    fancy_low.to_parquet(os.path.join(output_dir, "fancy_low.parquet"))
    fancy_high.to_parquet(os.path.join(output_dir, "fancy_high.parquet"))
    
    # Reload and verify the saved files
    clear_low_reloaded = pd.read_parquet(os.path.join(output_dir, "clear_low.parquet"))
    clear_high_reloaded = pd.read_parquet(os.path.join(output_dir, "clear_high.parquet"))
    fancy_low_reloaded = pd.read_parquet(os.path.join(output_dir, "fancy_low.parquet"))
    fancy_high_reloaded = pd.read_parquet(os.path.join(output_dir, "fancy_high.parquet"))
    
    pd.testing.assert_frame_equal(clear_low, clear_low_reloaded)
    pd.testing.assert_frame_equal(clear_high, clear_high_reloaded)
    pd.testing.assert_frame_equal(fancy_low, fancy_low_reloaded)
    pd.testing.assert_frame_equal(fancy_high, fancy_high_reloaded)

