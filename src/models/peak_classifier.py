import luigi
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

sys.path.insert(0, str(base_dir / "src" / "data"))
from common_functions import FW_GridSearchCV, get_most_recent_hyperparameters

class PeakClassifier(luigi.Task):
    outcome_selection = luigi.Parameter()
    hp_tuning = luigi.BoolParameter()
    hyperparameters = luigi.DictParameter() # only used if hp_tuning == True 

    def output(self):
        output_file_name = f"predicted_peaks_{self.outcome_selection}.pkl"
        return luigi.LocalTarget((base_dir/"data"/"interim"/output_file_name), format=luigi.format.Nop)
    
    def run(self):
        current_datetime = dt.datetime.now().strftime("%d%m%Y_%H%M")

        features_file = f"selected_features_{self.outcome_selection}_peak.pkl"

        def load_data(dataset_name):
            with open(base_dir/"data"/"interim"/"imputed_engineered_historical.pkl", "rb") as f:
                imported_data = pickle.load(f)
                data_all = imported_data['data']
                date_split = imported_data['date_split']
                taxa_list = imported_data['taxa_list']
                # Filter to dataset of choice
                data = data_all[dataset_name]
                return data, date_split, taxa_list

        import_data = load_data(self.outcome_selection)
        data = import_data[0] 
        date_split = import_data[1] 
        taxa_list = import_data[2] 

        with open(base_dir/"data"/"interim"/features_file, "rb") as f:
            imported_data = pickle.load(f)
            selected_features = imported_data["selected_features"]
            use_spore_lags = imported_data["use_spore_lags"]

        corrected_taxa_list = [item.replace("Total_fungal_spores", "sum_other_vars") for item in taxa_list]
        corrected_taxa_list = [taxa for taxa in corrected_taxa_list if taxa not in self.outcome_selection]

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

        scoring_methods = {"precision": "precision",
                "sens": make_scorer(sensitivity_score, greater_is_better=True, average='binary'),
                "spec": make_scorer(specificity_score, greater_is_better=True, average='binary'),
                "geometric": make_scorer(geometric_mean_score, greater_is_better=True, average='binary')
                }

        hyperparameter_results_directory = base_dir/"data"/"processed"/"hyperparameter_tuning"/self.outcome_selection

        try:
            recent_hyperparameters = get_most_recent_hyperparameters(hyperparameter_results_directory)
        except:
            recent_hyperparameters = 0


        if self.hp_tuning:
            grid_search_res = FW_GridSearchCV(features=train_X, labels=train_y, indice=train.index,
                                        model=EEC, parameter_grid=self.hyperparameters,
                                        scoring_methods=scoring_methods, rank_by="precision")
            EEC = EasyEnsembleClassifier(random_state=42, **grid_search_res[1])
            # Export results and best parameters
            results_table_filename = f"hp_tuning_peaks_results_{self.outcome_selection}_{current_datetime}.csv"
            results_best_parameters_filename = f"hp_tuning_peaks_{self.outcome_selection}_{current_datetime}.pkl"
            # Export tabular results
            grid_search_res[0].to_csv(base_dir/"data"/"processed"/"hyperparameter_tuning"/self.outcome_selection/results_table_filename)
            # Export best parameters
            with open(base_dir/"data"/"processed"/"hyperparameter_tuning"/self.outcome_selection/results_best_parameters_filename, "wb") as f:
                pickle.dump({"results": grid_search_res[1]}, f)
        elif recent_hyperparameters != 0:
            with open(base_dir/"data"/"processed"/"hyperparameter_tuning"/self.outcome_selection/recent_hyperparameters, "rb") as f:
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
        file_name = f"predicted_peaks_{self.outcome_selection}.pkl"

        with self.output().open("wb") as f:
            pickle.dump({'peaks': peaks_out,
                        "use_spore_lags": use_spore_lags}, f)