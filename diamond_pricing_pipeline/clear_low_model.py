import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_parquet("../data/diamonds.parquet")

# Column mappings
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
clarity_ord = ['F', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1',  'SI2', 'SI3', 'I1', 'I2','I3']
culet_size_ord = ['N', 'VS', 'S', 'M', 'SL', 'L', 'VL', 'EL', 'unknown']
cut_quality_ord = ['Ideal', 'Excellent',  'Very Good', 'Good', 'Fair', 'None', 'unknown']
polish_ord = ['Excellent','Very Good', 'Good', 'Fair', 'Poor']
symmetry_ord = ['Excellent','Very Good', 'Good', 'Fair', 'Poor']

ordinal_mappings = {
    'clarity': clarity_ord,
    'culet_size': culet_size_ord,
    'cut_quality': cut_quality_ord,
    'polish': polish_ord,
    'symmetry': symmetry_ord
}

nominal_columns = ['cut', 'color','lab','eye_clean','culet_condition',\
                   'girdle_max', 'girdle_min',\
                   'fancy_color_intensity','fancy_color_dominant_color',\
                   'fancy_color_secondary_color','fancy_color_overtone',\
                   'fluor_color', 'fluor_intensity']


# Preprocessing function for dataset
def preprocess_data(df, target_col='total_sales_price', test_size=0.2, val_size=0.25, random_state=1):
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train.pop(target_col)
    y_val = df_val.pop(target_col)
    y_test = df_test.pop(target_col)
    
    return df_train, df_val, df_test, y_train, y_val, y_test


# Break dataset into clear_diamonds and fancy_diamonds
clear_diamonds = df.loc[(df['fancy_color_dominant_color'] == 'unknown') & (df['color'] != 'unknown')]

fancy_diamonds = df.loc[(df['fancy_color_dominant_color'] != 'unknown') & (df['color'] == 'unknown')] 


# Split both datasets based on carat weight starting with clear:
clear_Q1_carat, clear_Q3_carat = clear_diamonds['carat_weight'].quantile(.25), clear_diamonds['carat_weight'].quantile(.75)
clear_IQR_carat = clear_Q3_carat - clear_Q1_carat
clear_upper_lim_carat = clear_Q3_carat + 1.5*clear_IQR_carat
clear_IQR_carat, clear_upper_lim_carat

clear_low_cw = clear_diamonds[clear_diamonds['carat_weight'] < clear_upper_lim_carat].copy()

clear_low_cw.loc[:, 'log_cw'] = np.log(clear_low_cw['carat_weight'])
clear_low_cw.loc[:, 'log_price'] = np.log(clear_low_cw['total_sales_price'])

clear_high_cw = clear_diamonds[clear_diamonds['carat_weight'] > clear_upper_lim_carat].copy()

clear_high_cw.loc[:, 'log_cw'] = np.log(clear_high_cw['carat_weight'])
clear_high_cw.loc[:, 'log_price'] = np.log(clear_high_cw['total_sales_price'])

print(f"The cutoff point is for clear diamonds is:  {clear_upper_lim_carat}")


# and now the fancy_diamonds 
# fancy_Q1_carat, fancy_Q3_carat = fancy_diamonds['carat_weight'].quantile(.25), fancy_diamonds['carat_weight'].quantile(.75)
# fancy_IQR_carat = fancy_Q3_carat - fancy_Q1_carat
# fancy_upper_lim_carat = fancy_Q3_carat + 1.5*fancy_IQR_carat
# fancy_IQR_carat, fancy_upper_lim_carat
#
# fancy_low_cw = fancy_diamonds[fancy_diamonds['carat_weight'] < fancy_upper_lim_carat].copy()
#
# fancy_low_cw.loc[:, 'log_cw'] = np.log(fancy_low_cw['carat_weight'])
# fancy_low_cw.loc[:, 'log_price'] = np.log(fancy_low_cw['total_sales_price'])
#
# print(f"The cutoff point is for fancy diamonds is:  {fancy_upper_lim_carat}")
#
# fancy_high_cw = fancy_diamonds[fancy_diamonds['carat_weight'] > fancy_upper_lim_carat].copy()
#
# fancy_high_cw.loc[:, 'log_cw'] = np.log(fancy_high_cw['carat_weight'])
# fancy_high_cw.loc[:, 'log_price'] = np.log(fancy_high_cw['total_sales_price'])
#










base = df[df['carat_weight'] < cutoff]
upper = df[(df['carat_weight'] >= cutoff) & (df['carat_weight'] <= 6.25)]

# Apply preprocessing for base and upper splits
clear_low_X_train, clear_low_X_val, clear_low_X_test, clear_low_y_train, clear_low_y_val, clear_low__y_test = preprocess_data(clear_low_cw)
clear_high_X_train, clear_high_X_val, clear_high_X_test, clear_high_y_train, clear_high_y_val, clear_high_y_test = preprocess_data(clear_high_cw)

# Pipeline for preprocessing
ordinal_transformers = [(col, OrdinalEncoder(categories=[categories]), [col]) for col, categories in ordinal_mappings.items()]
nominal_transformer = ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_columns)
numerical_columns = ['carat_weight', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth']
numerical_transformer = ('numerical', Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
]), numerical_columns)

preprocessor = ColumnTransformer(
    transformers=[*ordinal_transformers, nominal_transformer, numerical_transformer]
)

# Function to create model pipelines
def get_model_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Define XGBoost parameter grid for wider search
xgb_pipeline = get_model_pipeline(XGBRegressor(random_state=42))
xgb_params = {
    'model__n_estimators': [80, 90, 200, 225, 250, 275, 300],
    'model__max_depth': [8],
    'model__learning_rate': [0.15],
    'model__subsample': [.5, .55, .6, .65, .7, 0.75],
    'model__colsample_bytree': [.3]
}

# Run RandomizedSearchCV with neg_root_mean_squared_error scoring
xgb_random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_params,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_random_search.fit(baseX_train, base_y_train)

# Best model, parameters, and metrics
best_xgb_model = xgb_random_search.best_estimator_
best_params = xgb_random_search.best_params_
print(f"Best XGBoost parameters: {best_params}")


# def root_mean_squared_error(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred, squared=False)
#

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

print("Best Parameters for XGBoost Model:", xgb_random_search.best_params_)


# Updated evaluation function with explained variance
def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model using RMSE, R², and explained variance.
    """
    predictions = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, predictions)  # Updated to use sklearn's function
    r2 = r2_score(y_val, predictions)
    evs = explained_variance_score(y_val, predictions)  # Explained variance
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")
    print(f"Validation Explained Variance: {evs:.4f}")
    return rmse, r2, evs

# Evaluate the model on the base validation set
print("Evaluation Metrics for Best XGBoost Model:")
rmse, r2, evs = evaluate_model(best_xgb_model, baseX_val, base_y_val)

# Function to plot grid search results
def plot_grid_search_results(grid_search, param_grid):
    results = pd.DataFrame(grid_search.cv_results_)
    results['mean_test_score'] = -results['mean_test_score']  # Convert to positive RMSE

    for param in param_grid:
        if len(param_grid[param]) > 1:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=results, x=f'param_{param}', y='mean_test_score', marker='o')
            plt.title(f"Grid Search RMSE for {param}")
            plt.xlabel(param)
            plt.ylabel("Mean Test RMSE")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

# Call the plotting function for XGBoost results
plot_grid_search_results(xgb_random_search, xgb_params)

def plot_pca_variance(df, n_components=10):
    """
    Plot cumulative explained variance by principal components.
    """
    transformed_data = preprocessor.fit_transform(df).toarray()  # Convert to dense array
    pca = PCA(n_components=n_components)
    pca.fit(transformed_data)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Principal Components")
    plt.grid(True)
    plt.show()

# Plot PCA for base training set
plot_pca_variance(baseX_train)

importances = best_xgb_model.named_steps['model'].feature_importances_
feature_names = preprocessor.get_feature_names_out()
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x='Importance', y='Feature')
plt.title("Top 10 Feature Importances")
plt.show()



def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

# On validation set
preds = best_xgb_model.predict(baseX_val)
plot_residuals(base_y_val, preds)
