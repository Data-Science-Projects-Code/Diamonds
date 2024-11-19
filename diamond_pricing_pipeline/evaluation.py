from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

def build_preprocessing_pipeline(numerical_columns, ordinal_mappings, nominal_columns):
    """
    Constructs a preprocessing pipeline.
    
    Parameters:
    numerical_columns (list): List of numerical feature names.
    ordinal_mappings (dict): Dictionary with ordinal columns and their category order.
    nominal_columns (list): List of nominal feature names.
    
    Returns:
    ColumnTransformer: Preprocessing pipeline.
    """
    # All possible clarity categories 
    all_clarity_categories = ['I3', 'I2', 'I1', 'SI3', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'F']
    
    # Ensure the clarity column has all possible categories
    ordinal_transformers = [
        (
            col, 
            OrdinalEncoder(categories=[all_clarity_categories] if col == 'clarity' else [mapping], handle_unknown='use_encoded_value', unknown_value=-1),
            [col]
        ) 
        for col, mapping in ordinal_mappings.items()
    ]
    
    # Create nominal (one-hot) encoder
    nominal_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Standard scaler for numerical columns
    numerical_transformer = StandardScaler()
    
    # Now, combine all preprocessors
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_columns),
            ('ordinal', Pipeline(ordinal_transformers), list(ordinal_mappings.keys())),
            ('nominal', nominal_transformer, nominal_columns)
        ]
    )
    return preprocessor

