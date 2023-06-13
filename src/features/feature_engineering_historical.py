import luigi
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import statsmodels.api as sm
from scipy.stats import rankdata
from itertools import product
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIRECTORY')
base_dir = Path(base_dir)

sys.path.insert(0, str(base_dir / "src" / "data"))
from common_functions import (add_direction_class, add_days_since_meteorological_event, assign_site,
                              assign_decade_cat, add_temporal_features, growing_degree_days_group,
                              trim_seasons, peak_classify, incorporate_peaks, generate_lagged_features,
                              merge_non_target_spores, incorporate_clusters)


class FeatureEngineeringTrain(luigi.Task):
    lag_model = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget((base_dir/"data"/"interim"/"imputed_engineered_historical.pkl"), format=luigi.format.Nop)
    
    def run(self):
        # Import imputed time-series data
        with open(base_dir/"data"/"interim"/"imputed_historical.pkl", "rb") as f:
            imported_data = pickle.load(f)
            full = imported_data['data']
            imputation_markers = imported_data['imputation_markers']
            date_start = imported_data['date_start']
            date_split = imported_data['date_split']
            taxa_list = imported_data['taxa_list']

        # Split the time-series data
        train = full.query("date < @date_split")
        test = full.query("date >= @date_split")

        # Taxa not to be taken forward for modelling
        taxa_to_drop = ["Coloured_basidiospores", "Hyaline_basidiospores", "Other_spores", "Pleosporaceae"]
        train = train.drop(taxa_to_drop, axis=1)
        test = test.drop(taxa_to_drop, axis=1)

        # Keep an untouched train dataset (with annual outliers kept in), for use in feature generation
        full_untouched = pd.concat([train, test], axis=0).copy()

        # Outlier taxa-years
        # Import time-series feature derived outlier taxa-years references
        with open(base_dir/"data"/"interim"/"annual_outliers.pkl", "rb") as f:
            imported_data = pickle.load(f)
            outliers_derived = imported_data['outliers'].reset_index()

        # Import hand-selected outlier taxa-years references
        outliers_handselected = pd.read_csv(base_dir/"data"/"processed"/"forecast_years_to_remove.csv")

        # Merge outlier references and create a key column
        outliers = pd.concat([outliers_derived, outliers_handselected], axis=0).drop_duplicates()
        outliers["key"] = outliers["taxa"] + "_" + outliers["year"].astype("str")

        # Remove excluded taxa-years
        def remove_outliers(df):
            data = df.copy()
            data = pd.melt(data, id_vars=["date", "year"])
            data["key"] = data["variable"] + "_" + data["year"].astype("str")
            data = data[~data["key"].isin(outliers["key"])]
            data = data.reset_index()
            data = data.set_index("date")
            data = pd.pivot(data, columns="variable", values="value")
            return data

        train = remove_outliers(train)

        # Discretise wind direction variables
        add_direction_class(train, 'c_mean_wind_dir', 'c_mean_wind_dir_class')
        add_direction_class(train, 'c_max_gust_dir', 'c_max_gust_dir_class')
        add_direction_class(test, 'c_mean_wind_dir', 'c_mean_wind_dir_class')
        add_direction_class(test, 'c_max_gust_dir', 'c_max_gust_dir_class')

        # Import previously calculated meteorological thresholds
        #  note: these thresholds were determined with use of generalised additive modelling (GAM) of fungal spore counts
        #  as a smooth function of various individual meteorological variables. These thresholds were picked visually and 
        #  populated in the following csv file:
        meteo_thresholds = pd.read_csv(base_dir/"data"/"external"/"meteo_thresholds_GAM.csv")

        # Assign sampling site to observations as a feature
        train = assign_site(train)
        test = assign_site(test)

        # Assign decade integer as a feature
        train = assign_decade_cat(train)
        test = assign_decade_cat(test)

        # Add other temporal features (day of year, week, month)
        train = add_temporal_features(train)
        test = add_temporal_features(test)

        train = train.reset_index()
        date_split_dt = dt.datetime.combine(date_split, dt.datetime.min.time())

        # Convert the wind direction classes to a categorical varaible type
        train['c_mean_wind_dir_class'] = train['c_mean_wind_dir_class'].astype('category')
        train['c_max_gust_dir_class'] = train['c_max_gust_dir_class'].astype('category')

        # Group by site
        train_by_site = train[train['date'] < date_split_dt].groupby('site')
        # Find associations between taxa levels and wind directions (train only) with linear regression models,
        #  and reassign wind direction variables as ranked coefficients (enables multi-site use of wind direction)
        def wind_cor_table(site_selection, taxa_selection, variable_selection):
            dat = train_by_site.get_group(site_selection)
            formula = f"{taxa_selection} ~ {variable_selection}"
            model = sm.formula.ols(formula, data=dat).fit()
            summary = model.summary2().tables[1].reset_index()
            summary.columns = ['term', 'estimate', 'std.error', 't', 'P>|t|', '[0.025', '0.975]']
            intercept_name = f"{variable_selection}_1"
            summary.loc[summary['term'] == intercept_name, 'term'] = '1'
            summary['cor_rank'] = rankdata(-summary['estimate'])
            summary = summary.sort_values('cor_rank')
            summary['term'] = summary['term'].str.replace(variable_selection, '')
            summary['term'] = summary['term'].str.replace('\[T\.', '').str.replace('\]', '').str.replace('Intercept', '1')
            summary['taxa'] = taxa_selection
            summary['variable'] = variable_selection
            summary['site'] = site_selection
            summary = summary[['term', 'taxa', 'variable', 'site', 'cor_rank']]
            return summary

        # Create 'cross-table'  of site and taxa vectors
        sites_taxa_crossing = list(product(range(4), taxa_list))

        # Map the function to the site and taxa vectors
        wind_cor_tables_mean = [wind_cor_table(site, taxa, "c_mean_wind_dir_class") for site, taxa in sites_taxa_crossing]
        wind_cor_tables_max = [wind_cor_table(site, taxa, "c_max_gust_dir_class") for site, taxa in sites_taxa_crossing]

        # Combine the dataframes
        wind_cor_tables_mean = pd.concat(wind_cor_tables_mean, ignore_index=True)
        wind_cor_tables_max = pd.concat(wind_cor_tables_max, ignore_index=True)

        # Create reference dictionary
        wind_dir_cor_ref = {"c_mean_wind_dir_class": wind_cor_tables_mean, "c_max_gust_dir_class": wind_cor_tables_max}

        # Calculate fungal season parameters per fungal taxa and create a reference list 
        def calculate_fungal_seasons(taxa_selection, method=0.9):
            method_prop = (1-method)/2

            seasonal_ref = train[['date', taxa_selection]]
            seasonal_ref = seasonal_ref.rename(columns={taxa_selection: "value"})
            seasonal_ref['DoY'] = pd.DatetimeIndex(seasonal_ref['date']).dayofyear
            seasonal_ref['year'] = pd.DatetimeIndex(seasonal_ref['date']).year
            seasonal_ref['annual_sum'] = seasonal_ref.groupby('year')['value'].transform(np.sum)
            seasonal_ref['start_val'] = seasonal_ref['annual_sum'] * method_prop
            seasonal_ref['end_val'] = seasonal_ref['annual_sum'] * (1-method_prop)
            seasonal_ref['value_cumsum'] = seasonal_ref.groupby('year')['value'].transform(np.cumsum)

            seasonal_ref['seasonal_indice'] = np.where((seasonal_ref['value_cumsum'] >= seasonal_ref['start_val']) & (seasonal_ref['value_cumsum'] < seasonal_ref['end_val']), 1,
                                        np.where(seasonal_ref['value_cumsum'] >= seasonal_ref['end_val'], 2, 0))

            season_start = seasonal_ref[seasonal_ref['seasonal_indice'] == 1].groupby('year')['DoY'].min()
            season_end = seasonal_ref[seasonal_ref['seasonal_indice'] == 2].groupby('year')['DoY'].min()

            season_detail = pd.DataFrame({
                'season_start': season_start,
                'season_end': season_end
            }).reset_index()

            season_start_mean = round(season_detail['season_start'].mean(),0)
            season_end_mean = round(season_detail['season_end'].mean(),0)
            season_start_sd = round(season_detail['season_start'].std(),2)
            season_end_sd = round(season_detail['season_end'].std(),2)
            season_start_earliest = round(season_detail['season_start'].min(),0)
            season_end_latest = round(season_detail['season_end'].max(),0)

            season_detail_summary = pd.DataFrame({
                'taxa': taxa_selection,
                'season_start_mean': season_start_mean,
                'season_end_mean': season_end_mean,
                'season_start_sd': season_start_sd,
                'season_end_sd': season_end_sd,
                'season_start_earliest': season_start_earliest,
                'season_end_latest': season_end_latest,
                'years_used': len(season_detail)
            }, index=[0])

            return season_detail_summary, season_detail

        seasonal_dates_data = {taxa: calculate_fungal_seasons(taxa, method=0.9) for taxa in taxa_list}

        seasonal_dates = pd.concat([seasonal_dates_data[taxa][0] for taxa in taxa_list], ignore_index=True)

        # Calculate threshold concentrations per taxa for use in identifying peaks with peak_classify()
        def find_threshold(variable_selection, quantiles=7, topquantile=7):
            df = train[[variable_selection]].rename(columns={variable_selection: "variable"})
            df['variable_rank'] = df['variable'].rank(method='first')  # Rank the values uniquely (like ntile in R)
            df['quantile'] = pd.qcut(df['variable_rank'], q=quantiles, labels=False) + 1
            df = df[df['quantile'] == topquantile]
            threshold = round(df['variable'].min(), 0)
            y = pd.DataFrame({"variable": [variable_selection], "threshold": [threshold]})
            return y

        peak_thresholds_table = pd.concat([find_threshold(var) for var in taxa_list], ignore_index=True)

        peak_reference_table = {outcome: peak_classify(full_untouched, outcome, peak_thresholds_table, plot = False) for outcome in taxa_list}

        peaks_plots = {outcome: peak_reference_table[outcome]['plots'] for outcome in peak_reference_table.keys()}


        # Import day of year clusters (note: importing from R script output, still need to adapt for python)
        doy_clusters_spores_ref = pd.read_csv(base_dir/"data"/"interim"/"doy_clusters(fromR).csv")


        data = pd.concat([train, test.reset_index()], axis=0)

        # Identify the features to add lags to
        features_to_lag = ["c_avg_airpress", "c_avg_mean_wind_speed", "c_avg_relhum", "c_prcp_amt",
                        "c_gdd"]

        id_vars = ['date', 'site', 'decade_cat', 'year', 'month', 'week_1', 'DoY', 'c_avg_air_temp',
            'c_avg_airpress', 'c_avg_cloudcov', 'c_avg_dewpoint',
            'c_avg_mean_wind_speed', 'c_avg_relhum', 'c_max_air_temp',
            'c_max_gust_dir', 'c_max_mean_wind_speed', 'c_mean_wind_dir',
            'c_min_air_temp', 'c_prcp_amt', 'c_mean_wind_dir_class',
            'c_max_gust_dir_class']

        # Specify the columns to melt
        value_vars = ['Alternaria', 'AspPen', 'Botrytis', 'Cladosporium',
                    'Didymella', 'Drechslera', 'Entomophthora',
                    'Epicoccum', 'Erysiphe', 'Ganoderma',
                    'Leptosphaeria', 'Myxomycetes',
                    'Polythrincium', 'Rusts_smuts', 'Sporobolomyces', 'Tilletiopsis',
                    'Torula', 'Total_fungal_spores', 'Ustilago']

        # Perform the melt operation
        data_long = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')

        data_groups = data_long.groupby("variable")

        split_dataframes = {}
        for name, group in data_groups:
            # perform some operations on 'group' which is the split dataframe
            # The order of operations is important
            processed_group = group.sort_values(by="date")
            processed_group = growing_degree_days_group(processed_group, name, meteo_thresholds)
            processed_group = incorporate_clusters(processed_group, name, doy_clusters_spores_ref)
            processed_group = incorporate_peaks(processed_group, name, peak_reference_table)
            processed_group = generate_lagged_features(processed_group, features_to_lag, self.lag_model)
            processed_group = trim_seasons(processed_group, name, seasonal_dates)
            processed_group = add_days_since_meteorological_event(processed_group, name, meteo_thresholds)
            processed_group = processed_group.dropna(subset=['value'])
            processed_group = merge_non_target_spores(processed_group, name, full_untouched, value_vars, self.lag_model)
            processed_group = processed_group.iloc[self.lag_model:,]

            # store the processed group in the dictionary
            split_dataframes[name] = processed_group

        # Export
        with self.output().open('wb') as f:
            pickle.dump({'data': split_dataframes,
                        'date_start': date_start,
                        'date_split': date_split,
                        'taxa_list': taxa_list}, f)

