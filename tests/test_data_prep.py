import pytest
from diamond_pricing_pipeline.data_preparation import load_and_prepare_data

@pytest.fixture
def sample_file_path():
    return "data/diamonds.parquet"

def test_load_and_prepare_data(sample_file_path):
    clear, fancy = load_and_prepare_data(sample_file_path)
    assert len(clear) > 0, "Clear diamonds dataframe should not be empty."
    assert len(fancy) > 0, "Fancy diamonds dataframe should not be empty."
    assert "log_carat_weight" in clear.columns, "Clear diamonds should have 'log_carat_weight'."
    assert "log_total_sales_price" in fancy.columns, "Fancy diamonds should have 'log_total_sales_price'."

