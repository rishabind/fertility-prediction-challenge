"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcomes_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    # Combine cleaned_df and outcome_df
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcomes_df, on="nomem_encr")

     # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  

    # Upsampling
    children= model_df[model_df['new_child']==1]
    nochildren= model_df[model_df['new_child']==0]

    from sklearn.utils import resample
    children_upsample=resample(children, replace=True, n_samples=int(0.60*len(nochildren)), random_state=42)
    #print(children_upsample['new_child'].sum())

    data_upsampled= pd.concat([nochildren, children_upsample])
    #print(data_upsampled["new_child"].value_counts())
    
    
    ## Create imputer to impute missing values in the pipeline
    ## Create imputer to impute missing values in the pipeline
    
    imputer = KNNImputer(n_neighbors=2, weights="uniform").set_output(transform = "pandas")
    imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'constant',
                             fill_value = -1).set_output(transform ='pandas')

    ## Normalize variables
    numerical_columns = ["age", "birthyear_bg", "nettohh_f_2020"]
    categorical_columns = [ "cf20m003", "cf20m128", "cf20m013","cf20m024", "cf20m025",
                           "cf20m027","burgstat_2020", "oplmet_2020"]
    categorical_columns_ordinal = ["cf20m020", "cf20m129", "cf20m130", "cf20m022",
                                   "ci20m006","ci20m007","cv20l041","cv20l043","cv20l044","ci20m379"]
        
    categorical_preprocessor = make_pipeline(imputer2, OneHotEncoder(handle_unknown="ignore"))
    numerical_preprocessor = make_pipeline(imputer, StandardScaler())
    ordinal_preprocessor = make_pipeline(imputer2,
                                         OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value=-1))

    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns),
    ('ordinal_encoder', ordinal_preprocessor, categorical_columns_ordinal) ])
    
    # XG Boost model
    from sklearn.ensemble import GradientBoostingClassifier
    XG= make_pipeline(preprocessor,GradientBoostingClassifier())

    # Fit the model
    XG.fit(data_upsampled[["nomem_encr", "age", "woonvorm_2020","cf20m003",
                       "cf20m128", "cf20m129", "cf20m130", "birthyear_bg",
                       "nettohh_f_2020", "ci20m379", "cf20m013","cf20m020",
                       "cf20m022", "cf20m024", "cf20m025", "cf20m027", "cf20m030",
                       "burgstat_2020", "oplmet_2020","ci20m006","ci20m007",
                       "cv20l041","cv20l043","cv20l044"]], data_upsampled['new_child'])

    # Save the model
    joblib.dump(XG, "model_XG.joblib")