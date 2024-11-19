from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(numerical_columns, ordinal_mappings, nominal_columns):
    """
    Builds a preprocessing pipeline.
    
    Parameters:
    numerical_columns (list): List of numerical columns.
    ordinal_mappings (dict): Ordinal mappings for ordinal features.
    nominal_columns (list): List of nominal columns.
    
    Returns:
    ColumnTransformer: Preprocessing pipeline.
    """
    ordinal_transformers = [
        (col, OrdinalEncoder(categories=[categories]), [col])
        for col, categories in ordinal_mappings.items()
    ]
    nominal_transformer = ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_columns)
    numerical_transformer = ('numerical', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_columns)
    
    return ColumnTransformer(transformers=[*ordinal_transformers, nominal_transformer, numerical_transformer])

