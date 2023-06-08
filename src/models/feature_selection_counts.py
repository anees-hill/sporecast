#%%
import os
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIRECTORY')
base_dir = Path(base_dir)

outcome_selection = "Alternaria"

peaks_file = f"predicted_peaks_{outcome_selection}.pkl"

def load_data(dataset_name):
    with open(base_dir/"data"/"interim"/"imputed_engineered_historical.pkl", "rb") as f:
        imported_data = pickle.load(f)
        data_all = imported_data['data']
        date_split = imported_data['date_split']
        taxa_list = imported_data['taxa_list']
        # Filter to dataset of choice
        data = data_all[dataset_name]
        return data, date_split, taxa_list

import_data = load_data(outcome_selection)
data = import_data[0] 
date_split = import_data[1] 
taxa_list = import_data[2] 

with open(base_dir/"data"/"interim"/peaks_file, "rb") as f:
    imported_data = pickle.load(f)
    predicted_test_peaks = imported_data["peaks"]
    use_spore_lags = imported_data["use_spore_lags"]

corrected_taxa_list = [item.replace("Total_fungal_spores", "sum_other_vars") for item in taxa_list]
corrected_taxa_list = [taxa for taxa in corrected_taxa_list if taxa not in outcome_selection]

# Split
train = data.query("date < @date_split").set_index("date")
test = data.query("date >= @date_split").set_index("date")

# Make a copy of the relevant outcome for attachment to predicted 'peak' results for assessment purpose
outcome_counts_test = test["outcome"].copy()

# Remove cols not in use for modelling
train = train.drop(["outcome_lag1", "outcome_lag2", "outcome_lag3", "date_lag1", "date_lag2", "date_lag3",  "c_avg_air_temp", "c_min_air_temp", "c_mean_wind_dir", "c_max_gust_dir", "c_avg_cloudcov"], axis=1)
test = test.drop(["outcome_lag1", "outcome_lag2", "outcome_lag3", "date_lag1", "date_lag2", "date_lag3",  "c_avg_air_temp", "c_min_air_temp", "c_mean_wind_dir", "c_max_gust_dir", "c_avg_cloudcov"], axis=1)

if use_spore_lags == False:
    train = train.loc[:, ~train.columns.str.startswith(tuple(corrected_taxa_list))]
    test = test.loc[:, ~test.columns.str.startswith(tuple(corrected_taxa_list))]

# Separate outcome
train_X = train.drop("outcome", axis=1)
test_X = test.drop(["outcome", "peak"], axis=1)

# Make a copy of the outcome
train_y = train["outcome"].copy()
test_y = test["outcome"].copy()

# Add predicted peaks to test dataset
test_X["peak"] = predicted_test_peaks["predictions"]

# Separate cat and num cols
all_cols = train_X.columns.to_list()
categorical_cols = ["peak","month","site","decade_cat","week_1","year_type_for_doy_lag1","c_mean_wind_dir_class", "c_max_gust_dir_class"]
numerical_cols = [col for col in all_cols if col not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])
train_X_preprocessed = preprocessor.fit_transform(train_X)
test_X_preprocessed = preprocessor.transform(test_X)

# Get feature names from preprocessing
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)
num_feature_names = numerical_cols
feature_names = np.concatenate([ohe_feature_names, num_feature_names])

# Get transformed column names
column_names = preprocessor.get_feature_names_out()

# Convert preprocessed data back to dataframe
train_X_preprocessed_dense = train_X_preprocessed.toarray()
train_X_preprocessed_df = pd.DataFrame(train_X_preprocessed_dense, columns=column_names.tolist())

# Define Random Forest regressor
rf = RandomForestRegressor(n_jobs=-1, random_state=42, bootstrap=True, n_estimators=2000, max_depth=16, max_features=8, min_samples_leaf=1, min_samples_split=10)
# Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# Find relevant features
feat_selector.fit(train_X_preprocessed_df.values, train_y.values)

file_name = f"selected_features_{outcome_selection}_counts.pkl"

# Parse and export selected features
selected_features = train_X_preprocessed_df.columns[feat_selector.support_]
selected_features_clean = selected_features.str.replace('num__','').str.replace('cat__','')
with open(base_dir/"data"/"interim"/file_name, "wb") as f:
    pickle.dump({'selected_features': selected_features_clean,
                 "use_spore_lags": use_spore_lags}, f)

