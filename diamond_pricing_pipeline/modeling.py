from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def get_model_pipeline(preprocessor, model):
    """
    Wraps a model with a preprocessing pipeline.
    
    Parameters:
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    model: Machine learning model.
    
    Returns:
    Pipeline: Combined pipeline.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def tune_xgboost(X_train, y_train, preprocessor, random_state=42):
    """
    Tunes an XGBoost model using RandomizedSearchCV.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    Pipeline: Best XGBoost pipeline with tuned hyperparameters.
    """
    xgb_pipeline = get_model_pipeline(preprocessor, XGBRegressor(random_state=random_state))
    xgb_params = {
        'model__n_estimators': [80, 100, 150, 200, 250],
        'model__max_depth': [4, 6, 8],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__subsample': [0.6, 0.7, 0.8],
        'model__colsample_bytree': [0.3, 0.4, 0.5]
    }
    
    random_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_params,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    
    print(f"Best XGBoost parameters: {random_search.best_params_}")
    return random_search.best_estimator_

def train_linear_regression(X_train, y_train, preprocessor):
    """
    Trains a Linear Regression model with preprocessing.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    
    Returns:
    Pipeline: Trained Linear Regression pipeline.
    """
    lr_pipeline = get_model_pipeline(preprocessor, LinearRegression())
    lr_pipeline.fit(X_train, y_train)
    return lr_pipeline

if __name__ == "__main__":
    from data_prep import load_and_prepare_data
    from split_data import split_by_iqr
    from data_splits import preprocess_data
    from preprocessing_pipeline import build_preprocessing_pipeline
    
    file_path = "../data/diamonds.parquet"
    clear_diamonds, fancy_diamonds = load_and_prepare_data(file_path)
    clear_low, clear_high = split_by_iqr(clear_diamonds, column='log_carat_weight')
    clear_low_X_train, clear_low_X_val, clear_low_X_test, clear_low_y_train, clear_low_y_val, clear_low_y_test = preprocess_data(clear_low)
    
    numerical_columns = ['carat_weight', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth']
    ordinal_mappings = {
        'clarity': ['F', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'SI3', 'I1', 'I2', 'I3'],
        'cut_quality': ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'None', 'unknown']
    }
    nominal_columns = ['cut', 'color', 'lab', 'eye_clean']
    
    preprocessor = build_preprocessing_pipeline(numerical_columns, ordinal_mappings, nominal_columns)
    
    # Train models
    best_xgb_model = tune_xgboost(clear_low_X_train, clear_low_y_train, preprocessor)
    linear_model = train_linear_regression(clear_low_X_train, clear_low_y_train, preprocessor)
    
    print("Models trained successfully!")

