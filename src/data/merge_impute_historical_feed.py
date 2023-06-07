import luigi
import os
import pandas as pd
import numpy as np
from typing import Tuple
import datetime as dt
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
base_dir = os.getenv('BASE_DIRECTORY')
base_dir = Path(base_dir)


class MergeImputeHistoricalFeed(luigi.Task):
    ewm_span = luigi.Parameter()
    num_years = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget((base_dir/"data"/"interim"/"imputed_historical.pkl"), format=luigi.format.Nop)

    
    def run(self):
        # Read imported fungal spore counts and processed meteorological data
        spores = pd.read_csv(base_dir/"data"/"raw"/"daily_spores.csv")
        meteo = pd.read_csv(base_dir/"data"/"raw"/"meteo_midasfull_30kmMean.csv")

        # Formatting
        # Remove uneccesary column artefacts
        spores = spores.iloc[:,2:]
        # Correct date format
        spores["date"] = pd.to_datetime(spores["date"], dayfirst=True)
        spores["aero_type"] = spores["aero_type"].astype("str")

        # Select meteorological variables to utilise
        meteo = (meteo
                .loc[:,["date",
                        "c_avg_air_temp",
                        "c_avg_airpress",
                        "c_avg_cloudcov",
                        "c_avg_dewpoint",
                        "c_avg_mean_wind_speed",
                        "c_avg_relhum",
                        "c_max_air_temp",
                        "c_max_mean_wind_speed",
                        "c_min_air_temp",
                        "c_prcp_amt",
                        "c_max_gust_dir",
                        "c_mean_wind_dir"]]
        )

        meteo["date"] = pd.to_datetime(meteo["date"], dayfirst=True) # Recognise as dt
        meteo_vars = list(meteo.columns[1:])
        meteo_cat_vars = ["c_mean_wind_dir", "c_max_gust_dir"]

        # Filter for fungal taxa that meet criteria
        # These taxa will not neccesarily be outcomes but used in feature development
        response_selection = (spores
                        .assign(spores_present = lambda x: np.where(x['value'] < 1, 0, 1),
                                counted_day = lambda x: np.where(np.isnan(x['value'])==True,0,1))
                        .dropna()
                        .groupby(["variable"])
                        .agg(spores_present_sum=('spores_present', sum),
                            counted_day_sum=('counted_day', sum))
                        .assign(prevalence = lambda x: x["spores_present_sum"] / x["counted_day_sum"],
                                num_years = lambda x: x["counted_day_sum"]/365)
                        .reset_index()
                        .query("num_years > 27 & prevalence > 0.01 & variable != 'Unknown_spores'")
                        .sort_values(["num_years", "prevalence"], ascending=False)
                        .drop(["spores_present_sum","counted_day_sum"], axis = 1)
                        .round(2)
        )

        # Select fungal spore taxa for use in outcome (i.e. the models...)
        taxa_list = ["Ganoderma","Alternaria","AspPen","Botrytis",
                    "Cladosporium","Drechslera",
                    "Epicoccum","Erysiphe","Leptosphaeria","Polythrincium",
                    "Rusts_smuts","Sporobolomyces","Tilletiopsis","Torula",
                    "Ustilago","Didymella","Entomophthora","Myxomycetes","Total_fungal_spores"]

        # Fungal taxa only list
        response_selection_list = list(set(response_selection["variable"]))

        # Adapt fungal spore dataset to keep spores of interest
        spores = (spores
                    .query('variable in @response_selection_list')
                    .loc[:,["date","variable","value"]]
                    .set_index("date")
                    .pivot(columns="variable", values="value")
                    .reset_index()
        )
        # spores["date"] = spores["date"].dt.date # Keep date only (not time)

        # Fix column names for spores
        columns_spores = spores.columns
        columns_spores_stripped = [name.strip() for name in columns_spores]
        spores.columns = columns_spores_stripped

        # Define start date (1970, to avoid pollen only years)
        date_start = dt.datetime.strptime("01-01-1970", "%d-%m-%Y").date() # .date() gets rid of the time part

        # Define train_test split date
        date_split = dt.datetime.strptime("01-01-2016", "%d-%m-%Y").date()

        # Merge fungal spore and meteorological data
        spores_meteo = (spores
                .merge(meteo, how="outer", on="date")
                .query("date >= @date_start")
        )

        # Insert 'year' column for imputation
        spores_meteo["year"] = spores_meteo["date"].dt.year

        # Split datasets for imputation
        train = spores_meteo.copy()
        test_train = spores_meteo.copy()

        # Remove 'test' subset from train
        train = train.query("date < @date_split")

        # Define imputation groups
        cols_to_impute1 = ["date","year",
                        "c_avg_air_temp", "c_avg_airpress", "c_avg_cloudcov", "c_avg_dewpoint",
                        "c_avg_mean_wind_speed", "c_avg_relhum", "c_max_air_temp", "c_max_mean_wind_speed",
                            "c_min_air_temp", "c_prcp_amt"]

        cols_to_impute2 = ["date","year",
                        "c_mean_wind_dir", "c_max_gust_dir"]

        cols_to_impute3 = ["date","year",
                        'Coloured_basidiospores', 'Ganoderma','Hyaline_basidiospores', 'Alternaria',
                        'AspPen', 'Botrytis', 'Total_fungal_spores', 'Cladosporium', 'Didymella',
                            'Drechslera', 'Entomophthora', 'Epicoccum', 'Erysiphe', 'Leptosphaeria',
                            'Pleosporaceae', 'Myxomycetes', 'Other_spores', 'Polythrincium', 'Rusts_smuts',
                                'Sporobolomyces', 'Tilletiopsis', 'Torula', 'Ustilago']

        # For use in impute_data(). Finds the nearest n years for the observation being processed
        def get_nearest_years(df: pd.DataFrame, target_year: int, column: str, month: int, day: int, num_years: int) -> pd.Series:
            """Gets the nearest num_years of data (or less if not available) for a specific date and column."""
            sorted_years = df.loc[(df.date.dt.month == month) & (df.date.dt.day == day)].year.subtract(target_year).abs().sort_values()
            nearest_years = sorted_years.head(num_years).index
            return df.loc[nearest_years, column]

        # Impute missing observations in a 3-step process: (1) EWM, (2) Average for day of year acorss the nearest years,
        #  (3) backfill any remaining NULLS (found to be the first few zero values)
        def impute_data(df: pd.DataFrame, ewm_span: int = 3, num_years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
            df_copy = df.copy()
            imputed = pd.DataFrame(False, index=df.index, columns=df.columns)

            for col in df.columns:
                if df[col].dtype == np.object or col in ['date', 'year']:
                    continue

                # Fill nulls with EWM
                ema = df_copy[col].ewm(span=ewm_span, adjust=False).mean()
                df_copy[col].fillna(ema, inplace=True)
                # Record where the filled values were replaced with EWM
                mask_ewm_filled = df_copy[col].isnull()

                # For remaining nulls: fill average of the same day of the year across the nearest num_years
                mask_remaining_nulls = df_copy[col].isnull()
                for idx in df_copy[mask_remaining_nulls].index:
                    row = df_copy.loc[idx]
                    year = row.year
                    month = row.date.month
                    day = row.date.day

                    nearest_years_data = get_nearest_years(df_copy, year, col, month, day, num_years)
                    df_copy.loc[idx, col] = nearest_years_data.mean()

                    # Mark these values as imputed in imputed df
                    imputed.loc[idx, col] = True

                # Mark EWM filled values as imputed in imputed df
                imputed.loc[mask_ewm_filled, col] = True

                # For remaining nulls, backfill
                mask_pre_backfill = df_copy[col].isnull() 
                df_copy[col].fillna(method='bfill', inplace=True)

                # Mark backfilled values as imputed in imputed df
                mask_post_backfill = df_copy[col].notna()
                imputed.loc[mask_pre_backfill & mask_post_backfill, col] = True

            return df_copy, imputed


        # Run impute_data() for each dataset and variable category (spores / meteorological numerical)
        train_imputed_1, train_imputed_1_markers = impute_data(train.loc[:,cols_to_impute1], ewm_span=5, num_years=12)
        train_imputed_2, train_imputed_2_markers = impute_data(train.loc[:,cols_to_impute2], ewm_span=5, num_years=12)
        train_imputed_3, train_imputed_3_markers = impute_data(train.loc[:,cols_to_impute3], ewm_span=7, num_years=12)

        test_train_imputed_1, test_train_imputed_1_markers = impute_data(test_train.loc[:,cols_to_impute1], ewm_span=5, num_years=12)
        test_train_imputed_2, test_train_imputed_2_markers = impute_data(test_train.loc[:,cols_to_impute2], ewm_span=5, num_years=12)
        test_train_imputed_3, test_train_imputed_3_markers = impute_data(test_train.loc[:,cols_to_impute3], ewm_span=7, num_years=12)

        # merge with training set
        train_postimpute = pd.merge(train_imputed_1, train_imputed_2, on=['date','year'])
        train_postimpute = pd.merge(train_postimpute, train_imputed_3, on=['date','year'])

        # Merge test-train
        test_train_postimpute = pd.merge(test_train_imputed_1, test_train_imputed_2, on=['date','year'])
        test_train_postimpute = pd.merge(test_train_postimpute, test_train_imputed_3, on=['date','year'])

        # Keep test section only
        test_train_postimpute = test_train_postimpute.query("date >= @date_split")

        # Merge all
        full_postimpute = pd.concat([train_postimpute, test_train_postimpute], axis=0)

        # Group together imputation markers (concat as there is no 'date' column)
        train_imputed_markers = pd.concat([train_imputed_1_markers, train_imputed_2_markers, train_imputed_3_markers], axis = 1)
        test_train_imputed_markers = pd.concat([test_train_imputed_1_markers, test_train_imputed_2_markers, test_train_imputed_3_markers], axis = 1)

        imputed_markers = {
            "train": train_imputed_markers,
            "test_train": test_train_imputed_markers,
        }

        # Export
        with self.output().open('wb') as f:
            pickle.dump({'data': full_postimpute,
                        'imputation_markers': imputed_markers,
                        'date_start': date_start,
                        'date_split': date_split,
                        'taxa_list': taxa_list}, f)







