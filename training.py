"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(data, outcome_df):

 ## Preprocess the data
    categorical_columns_ordinal = ["cf20m181","ci20m006","ci20m007", "cv20l041","cv20l043","cv20l044"]
    categorical_columns_onehot =["cf20m003","cf20m030", "ci20m008"]
    numeric_columns =["ch20m002"]
    
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder().set_output(transform="pandas")
    
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()
    
    from sklearn.compose import ColumnTransformer
    
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns_onehot),
        ('standard_scaler', numerical_preprocessor, numeric_columns)], remainder="passthrough")
       
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    
    #  model
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  

    # Fit the model
    model.fit(model_df[['age']], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")
