import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='log_total_sales_price', test_size=0.2, val_size=0.25, random_state=1):
    """
    Splits a single dataset into train, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The dataframe to split.
    - target_col (str): The column to predict.
    - test_size (float): Proportion of data for test set.
    - val_size (float): Proportion of training data for validation set.
    - random_state (int): Random state for reproducibility.

    Returns:
    - X_train, X_val, X_test (pd.DataFrame): Features for train, validation, and test sets.
    - y_train, y_val, y_test (pd.Series): Targets for train, validation, and test sets.
    """
    # Split into train+validation and test
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    # Split train+validation into train and validation
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Separate features and target
    y_train = df_train.pop(target_col)
    y_val = df_val.pop(target_col)
    y_test = df_test.pop(target_col)

    return df_train, df_val, df_test, y_train, y_val, y_test


def split_and_display_shapes(dataframes, target_col='log_total_sales_price', test_size=0.2, val_size=0.25, random_state=1):
    """
    Splits multiple datasets into train, validation, and test sets and prints their shapes.

    Parameters:
    - dataframes (dict): A dictionary of named dataframes to split.
    - target_col (str): The column to predict in each dataframe.
    - test_size (float): Proportion of data for test set.
    - val_size (float): Proportion of training data for validation set.
    - random_state (int): Random state for reproducibility.

    Returns:
    - splits (dict): A dictionary containing train, validation, and test splits for each dataframe.
    """
    splits = {}
    for name, dataframe in dataframes.items():
        # Preprocess each dataframe
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            dataframe,
            target_col=target_col,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        # Save the splits
        splits[name] = {
            f"{name}_X_train": X_train,
            f"{name}_X_val": X_val,
            f"{name}_X_test": X_test,
            f"{name}_y_train": y_train,
            f"{name}_y_val": y_val,
            f"{name}_y_test": y_test,
        }

        # Output concise shapes
        print(
            f"{name.replace('_', ' ').capitalize()} - "
            f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}"
        )

    return splits


if __name__ == "__main__":
    # Load your actual datasets here
    clear_low = pd.read_parquet('../data/splits/clear_low.parquet')
    clear_high = pd.read_parquet('../data/splits/clear_high.parquet')
    fancy_low = pd.read_parquet('../data/splits/fancy_low.parquet')
    fancy_high = pd.read_parquet('../data/splits/fancy_high.parquet')

    datasets = {
        "clear_low": clear_low,
        "clear_high": clear_high,
        "fancy_low": fancy_low,
        "fancy_high": fancy_high,
    }

    # Perform splits and display shapes
    splits = split_and_display_shapes(datasets)

