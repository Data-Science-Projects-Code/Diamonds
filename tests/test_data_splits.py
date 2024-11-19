import pytest
import pandas as pd
from data_splits import preprocess_data, split_all_dataframes


@pytest.fixture
def example_datasets():
    """
    Fixture to create example datasets for testing.
    """
    clear_low_df = pd.DataFrame({"feature1": range(100), "log_total_sales_price": range(100)})
    clear_high_df = pd.DataFrame({"feature1": range(50), "log_total_sales_price": range(50)})
    fancy_low_df = pd.DataFrame({"feature1": range(75), "log_total_sales_price": range(75)})
    fancy_high_df = pd.DataFrame({"feature1": range(25), "log_total_sales_price": range(25)})

    datasets = {
        "clear_low": clear_low_df,
        "clear_high": clear_high_df,
        "fancy_low": fancy_low_df,
        "fancy_high": fancy_high_df,
    }

    return datasets


def test_preprocess_data():
    """
    Test the preprocess_data function with a sample dataframe.
    """
    # Create a sample dataframe
    df = pd.DataFrame({"feature1": range(100), "log_total_sales_price": range(100)})
    target_col = "log_total_sales_price"

    # Run the function
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target_col)

    # Assertions
    assert len(X_train) + len(X_val) + len(X_test) == len(df), "Split sizes do not match original dataset size"
    assert X_train.shape[1] == df.shape[1] - 1, "Feature count mismatch after splitting"
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == len(df), "Target sizes do not match original dataset size"


def test_split_all_dataframes(example_datasets):
    """
    Test the split_all_dataframes function with multiple datasets.
    """
    splits = split_all_dataframes(example_datasets)

    # Check if all datasets were split
    for name, dataset in example_datasets.items():
        assert name in splits, f"{name} splits are missing from the output"
        
        split_data = splits[name]
        X_train, X_val, X_test = (
            split_data[f"{name}_X_train"],
            split_data[f"{name}_X_val"],
            split_data[f"{name}_X_test"],
        )
        y_train, y_val, y_test = (
            split_data[f"{name}_y_train"],
            split_data[f"{name}_y_val"],
            split_data[f"{name}_y_test"],
        )

        # Verify split sizes
        assert len(X_train) + len(X_val) + len(X_test) == len(dataset), f"{name}: Split sizes do not match dataset size"
        assert X_train.shape[1] == dataset.shape[1] - 1, f"{name}: Feature count mismatch after splitting"
        assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == len(dataset), f"{name}: Target sizes do not match dataset size"

        # Verify train/validation/test ratio
        total_samples = len(dataset)
        test_size = int(total_samples * 0.2)
        train_val_size = total_samples - test_size
        val_size = int(train_val_size * 0.25)
        train_size = train_val_size - val_size

        assert len(X_train) == train_size, f"{name}: Train size mismatch"
        assert len(X_val) == val_size, f"{name}: Validation size mismatch"
        assert len

