import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from tqdm import tqdm
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Load data
df = pd.read_parquet("../data/diamonds.parquet")

# Define column lists and mappings
numerical_columns = ['carat_weight', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth']
clarity_ord = ['F', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'SI3', 'I1', 'I2', 'I3']
culet_size_ord = ['N', 'VS', 'S', 'M', 'SL', 'L', 'VL', 'EL', 'unknown']
cut_quality_ord = ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'None', 'unknown']
polish_ord = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
symmetry_ord = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']

ordinal_mappings = {
    'clarity': clarity_ord,
    'culet_size': culet_size_ord,
    'cut_quality': cut_quality_ord,
    'polish': polish_ord,
    'symmetry': symmetry_ord
}
nominal_columns = ['cut', 'color', 'lab', 'eye_clean', 'culet_condition', 'girdle_max', 'girdle_min',
                   'fancy_color_intensity', 'fancy_color_dominant_color', 'fancy_color_secondary_color',
                   'fancy_color_overtone', 'fluor_color', 'fluor_intensity']

# Split data into base and upper based on carat weight
Q1_carat_weight, Q3_carat_weight = df['carat_weight'].quantile(.25), df['carat_weight'].quantile(.75)
IQR_carat_weight = Q3_carat_weight - Q1_carat_weight
cutoff = Q3_carat_weight + 1.5 * IQR_carat_weight

base = df[df['carat_weight'] < cutoff]
upper = df[df['carat_weight'] >= cutoff]

# Preprocessing function for training/validation/test split
def preprocess_data(df, target_col='total_sales_price', test_size=.2, val_size=0.25, random_state=1):
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    y_train = df_train.pop(target_col)
    y_val = df_val.pop(target_col)
    y_test = df_test.pop(target_col)
    return df_train, df_val, df_test, y_train, y_val, y_test

# Train-validation-test split for base and upper datasets
baseX_train, base_val, base_test, base_y_train, base_y_val, base_y_test = preprocess_data(base, target_col='total_sales_price')
upperX_train, upper_val, upper_test, upper_y_train, upper_y_val, upper_y_test = preprocess_data(upper, target_col='total_sales_price')

# Preprocessing pipeline
ordinal_transformers = [(col, OrdinalEncoder(categories=[categories]), [col]) for col, categories in ordinal_mappings.items()]
nominal_transformer = ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_columns)
numerical_transformer = ('numerical', Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
]), numerical_columns)

preprocessor = ColumnTransformer(
    transformers=[
        *ordinal_transformers,
        nominal_transformer,
        numerical_transformer
    ]
)

# Pipeline for model training
def get_model_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Define models and grid search parameters
models = {
    #'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(random_state=42)
}

param_grids = {
    # 'Linear Regression': {},
    'XGBoost': {
        'model__n_estimators': [200, 300, 350],
        'model__max_depth': [10, 11, 12],
        'model__learning_rate': [0.01, 0.03],
        'model__subsample': [0.4, 0.5,0.6]
    }
}

# Model evaluation function
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    return {"MSE": mse, "RMSE": rmse, "R-squared": r2}

# Train each model with grid search and evaluate
best_models = {}
best_grid_searches = {}
metrics_summary = {}

for model_name, model in tqdm(models.items(), desc="Models"):
    pipeline = get_model_pipeline(model)
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(baseX_train, base_y_train)
    best_grid_searches[model_name] = grid_search
    best_models[model_name] = grid_search.best_estimator_
    metrics_summary[model_name] = evaluate_model(best_models[model_name], base_val, base_y_val)
    if hasattr(grid_search, 'best_params_'):
        print(f"\nBest Parameters for {model_name}:")
        print(grid_search.best_params_)

# Summarize evaluation metrics
evaluation_summary_df = pd.DataFrame(metrics_summary).T
evaluation_summary_df.index.name = 'Model'
print("\nEvaluation Summary:")
print(evaluation_summary_df)

# Save the best model for future use
joblib.dump(best_models['XGBoost'], 'best_xgboost_model.pkl')

# Plot grid search results for XGBoost
def plot_grid_search_results(grid_search, param_grid):
    results = pd.DataFrame(grid_search.cv_results_)
    for param in param_grid:
        if len(param_grid[param]) > 1:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=results, x=f'param_{param}', y='mean_test_score', marker='o')
            plt.title(f"Grid Search Results for {param}")
            plt.xlabel(param)
            plt.ylabel("Mean Test Score (Negative MSE)")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

plot_grid_search_results(best_grid_searches['XGBoost'], param_grids['XGBoost'])

# Cross-validation score visualization
def plot_cv_scores(grid_search):
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['RMSE'] = cv_results['RMSE']
    fold_scores = cv_results[[f'split{i}_test_score' for i in range(grid_search.cv)]]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=-fold_scores)
    plt.title("Cross-Validation Scores Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("Root Mean Squqred Area(RMSE)")
    plt.grid(True)
    plt.show()

plot_cv_scores(best_grid_searches['XGBoost'])

