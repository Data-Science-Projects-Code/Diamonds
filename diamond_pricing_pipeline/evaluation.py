import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model using RMSE, R², and explained variance.
    
    Parameters:
    model (Pipeline): Trained pipeline model.
    X_val (DataFrame): Validation features.
    y_val (Series): Validation target.
    
    Returns:
    tuple: (RMSE, R², Explained Variance)
    """
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)
    evs = explained_variance_score(y_val, predictions)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")
    print(f"Validation Explained Variance: {evs:.4f}")
    return rmse, r2, evs

def plot_feature_importances(model, preprocessor):
    """
    Plot the top 10 feature importances for an XGBoost model.
    
    Parameters:
    model (Pipeline): Trained XGBoost pipeline.
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    """
    feature_importances = model.named_steps['model'].feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feature_importance_df = (
        pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        .sort_values(by='Importance', ascending=False)
    )
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature')
    plt.title("Top 10 Feature Importances")
    plt.show()

def plot_residuals(y_true, y_pred):
    """
    Plot residuals for model predictions.
    
    Parameters:
    y_true (Series): True target values.
    y_pred (Series): Predicted values.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    from modeling import tune_xgboost
    from preprocessing_pipeline import build_preprocessing_pipeline
    from data_splits import preprocess_data
    from split_data import split_by_iqr
    from data_prep import load_and_prepare_data

    file_path = "../data/diamonds.parquet"
    clear_diamonds, fancy_diamonds = load_and_prepare_data(file_path)
    clear_low, _ = split_by_iqr(clear_diamonds, column='log_carat_weight')
    clear_low_X_train, clear_low_X_val, clear_low_X_test, clear_low_y_train, clear_low_y_val, clear_low_y_test = preprocess_data(clear_low)
    
    preprocessor = build_preprocessing_pipeline(
        numerical_columns=['carat_weight', 'depth_percent', 'table_percent'],
        ordinal_mappings={
            'clarity': ['F', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'SI3']
        },
        nominal_columns=['cut', 'color']
    )
    best_xgb_model = tune_xgboost(clear_low_X_train, clear_low_y_train, preprocessor)
    evaluate_model(best_xgb_model, clear_low_X_val, clear_low_y_val)

