#%%
import os
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIRECTORY')
base_dir = Path(base_dir)

current_datetime = dt.datetime.now().strftime("%d%m%Y_%H%M")

# Externally declared parameters
# Select taxa to model
outcome_selection = "Alternaria"
# Select whether to tune hyperparameters
hp_tuning = False

features_file = f"selected_features_{outcome_selection}_peak.pkl"

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

with open(base_dir/"data"/"interim"/features_file, "rb") as f:
    imported_data = pickle.load(f)
    selected_features = imported_data["selected_features"]
    use_spore_lags = imported_data["use_spore_lags"]

corrected_taxa_list = [item.replace("Total_fungal_spores", "sum_other_vars") for item in taxa_list]
corrected_taxa_list = [taxa for taxa in corrected_taxa_list if taxa not in outcome_selection]

# Split
train = data.query("date < @date_split").set_index("date")
test = data.query("date >= @date_split").set_index("date")

# Make a copy of the relevant outcome for attachment to predicted 'peak' results for assessment purpose
outcome_counts_test = test["outcome"].copy()

# Remove cols not in use for modelling
train = train.drop(["outcome", "outcome_lag1", "outcome_lag2", "outcome_lag3", "date_lag1", "date_lag2", "date_lag3",  "c_avg_air_temp", "c_min_air_temp", "c_mean_wind_dir", "c_max_gust_dir", "c_avg_cloudcov"], axis=1)
test = test.drop(["outcome", "outcome_lag1", "outcome_lag2", "outcome_lag3", "date_lag1", "date_lag2", "date_lag3",  "c_avg_air_temp", "c_min_air_temp", "c_mean_wind_dir", "c_max_gust_dir", "c_avg_cloudcov"], axis=1)

if use_spore_lags == False:
    train = train.loc[:, ~train.columns.str.startswith(tuple(corrected_taxa_list))]
    test = test.loc[:, ~test.columns.str.startswith(tuple(corrected_taxa_list))]


# Separate labels
train_X = train.drop("peak", axis=1)
test_X = test.drop("peak", axis=1)

# Make a copy of the labels/convert to integer
train_y = train["peak"].copy().astype(int)
test_y = test["peak"].copy().astype(int)


# Separate cat and num cols
all_cols = train_X.columns.to_list()
categorical_cols = ["month","site","decade_cat","week_1","year_type_for_doy_lag1","c_mean_wind_dir_class", "c_max_gust_dir_class"]
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

# Check `selected_features` is present, then find the indices of the previously `selected_features`
if 'selected_features' not in locals():
    raise NameError("selected_features is not defined")
selected_indices = [i for i, name in enumerate(feature_names) if name in selected_features]

# Select only the columns of the preprocessed data corresponding to the selected features
train_X = train_X_preprocessed[:, selected_indices]
test_X = test_X_preprocessed[:, selected_indices]

# Define a classifier for use in grid search
EEC = EasyEnsembleClassifier(random_state=42, sampling_strategy=0.1, n_estimators=100)

# Set tuning parameters (only used if `hp_tuning=True`)
tuning_parameters = [
     {'n_estimators': [3], 'sampling_strategy': [0.1]},
  ]

scoring_methods = {"precision": "precision",
           "sens": make_scorer(sensitivity_score, greater_is_better=True, average='binary'),
           "spec": make_scorer(specificity_score, greater_is_better=True, average='binary'),
           "geometric": make_scorer(geometric_mean_score, greater_is_better=True, average='binary')
          }

def FW_GridSearchCV(features, labels, indice, model, parameter_grid, scoring_methods, rank_by):
    # Convert features to array and create a DataFrame
    input_array = features.toarray()
    input_dataframe = pd.DataFrame(input_array, index=indice)
    input_dataframe['year'] = input_dataframe.index.year
    input_dataframe.columns = input_dataframe.columns.astype(str)

    # Get unique years
    unique_years = sorted(input_dataframe['year'].unique())
    n_splits = min(9, len(unique_years))
    years_per_split = len(unique_years) // n_splits

    training_sets = {}
    training_labels = {}
    tscv = {}
    grid_search = {}

    best_score = float('-inf')
    best_params = None

    for i in range(n_splits):
        # Calculate the start year for each split
        start_year = unique_years[i * years_per_split]

        # Extract indexes for each year
        ind = input_dataframe.query(f'year >= {start_year}').index

        # Filter training set and labels
        training_sets[start_year] = input_dataframe.loc[ind]
        training_labels[start_year] = labels.loc[ind]

        # Define time-series split
        tscv[start_year] = TimeSeriesSplit(n_splits=i + 2, max_train_size=None, test_size=None, gap=0)

        # Define GridSearchCV
        grid_search[start_year] = GridSearchCV(estimator=model, cv=tscv[start_year], param_grid=parameter_grid, scoring=scoring_methods,
                                               verbose=3, return_train_score=True, refit=False, n_jobs=1)

        # Fit the model
        grid_search[start_year].fit(training_sets[start_year], training_labels[start_year])

        # Check if this model has better score than previous best
        current_best_index = grid_search[start_year].cv_results_['rank_test_' + rank_by].argmin()
        current_best_score = grid_search[start_year].cv_results_['mean_test_' + rank_by][current_best_index]

        if current_best_score > best_score:
            best_score = current_best_score
            best_params = grid_search[start_year].cv_results_['params'][current_best_index]

    # Collect results in a DataFrame
    cvres_pd = pd.concat([pd.DataFrame(grid_search[year].cv_results_) for year in training_sets.keys()])

    # Add date run
    cvres_pd["date_run"] = pd.Series([dt.datetime.now()] * len(cvres_pd))

    # Rank by selected metric
    cvres_pd = cvres_pd.sort_values(by=f'mean_test_{rank_by}', ascending=False)

    return cvres_pd, best_params



def get_most_recent_hyperparameters(directory):
    # Get list of files in the directory
    files = os.listdir(directory)

    # Keep only files that end with '.pkl'
    pkl_files = [file for file in files if file.endswith('.pkl')]

    # Extract numbers from filenames and convert to datetime
    datetimes = {}
    for file in pkl_files:
        # Extract numbers
        numbers = ''.join(ch for ch in file if ch.isdigit())
        # Convert to datetime
        dt_x = dt.datetime.strptime(numbers, '%d%m%Y%H%M')
        datetimes[dt_x] = file

    # Get the filename corresponding to the most recent date
    max_dt = max(datetimes.keys())
    most_recent_file = datetimes[max_dt]

    return most_recent_file

hyperparameter_dir = base_dir/"data"/"processed"/"hyperparameter_tuning"/outcome_selection

try:
    recent_hyperparameters = get_most_recent_hyperparameters(hyperparameter_dir)
except:
    recent_hyperparameters = 0


if hp_tuning:
    grid_search_res = FW_GridSearchCV(features=train_X, labels=train_y, indice=train.index,
                                  model=EEC, parameter_grid=tuning_parameters,
                                  scoring_methods=scoring_methods, rank_by="precision")
    EEC = EasyEnsembleClassifier(random_state=42, **grid_search_res[1])
    # Export results and best parameters
    results_table_filename = f"hp_tuning_peaks_results_{outcome_selection}_{current_datetime}.csv"
    results_best_parameters_filename = f"hp_tuning_peaks_{outcome_selection}_{current_datetime}.pkl"
    # Export tabular results
    grid_search_res[0].to_csv(base_dir/"data"/"processed"/"hyperparameter_tuning"/outcome_selection/results_table_filename)
    # Export best parameters
    with open(base_dir/"data"/"processed"/"hyperparameter_tuning"/outcome_selection/results_best_parameters_filename, "wb") as f:
        pickle.dump({"results": grid_search_res[1]}, f)
elif recent_hyperparameters != 0:
    with open(base_dir/"data"/"processed"/"hyperparameter_tuning"/outcome_selection/recent_hyperparameters, "rb") as f:
        imported_data = pickle.load(f)
        best_params = imported_data["results"]
    EEC = EasyEnsembleClassifier(random_state=42, **best_params)
    EEC.fit(train_X, train_y)
else:
    print("Default model parameters in use")
    EEC = EasyEnsembleClassifier(random_state=42)
    EEC.fit(train_X, train_y)


peak_predictions = EEC.predict(test_X)
EEC_test_precision = precision_score(test_y, peak_predictions)
print(EEC_test_precision)


# PREDICTIONS COMPONENT (select train or test)
peaks_out = pd.DataFrame(peak_predictions, columns = ["predictions"], index=test_y.index)

# Attach calculated peaks for model assessment
peaks_out = pd.concat([peaks_out, test_y], axis=1)
peaks_out = pd.concat([peaks_out, outcome_counts_test], axis=1)

# Export peaks
file_name = f"predicted_peaks_{outcome_selection}.pkl"

with open(base_dir/"data"/"interim"/file_name, "wb") as f:
    pickle.dump({'peaks': peaks_out,
                 "use_spore_lags": use_spore_lags}, f)


