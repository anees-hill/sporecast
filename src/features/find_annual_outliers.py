import luigi
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import kats
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIRECTORY')
base_dir = Path(base_dir)


class FindAnnualOutliers(luigi.Task):

    def output(self):
        return luigi.LocalTarget((base_dir/"data"/"interim"/"annual_outliers.pkl"), format=luigi.format.Nop)
    
    def run(self):
        # Import imputed data and relevant parameters
        with open(base_dir/"data"/"interim"/"imputed_historical.pkl", "rb") as f:
            imported_data = pickle.load(f)
            full = imported_data['data']
            imputation_markers = imported_data['imputation_markers']
            date_start = imported_data['date_start']
            date_split = imported_data['date_split']
            taxa_list = imported_data['taxa_list']

        # Use previously defined split date to isolate train set
        train = full.query("date < @date_split")

        # List of all time-series features possible for extraction
        pca_all_avail_factors = ["mean", "var", "entropy", "lumpiness", "stability", "flat_spots", "hurst", "std1st_der",
                                "crossing_points", "binarize_mean", "unitroot_kpss", "heterogeneity", "histogram_mode", "linearity",
                                    "trend_strength", "spikiness", "peak", "trough", "level_shift_idx", "level_shift_size", "y_acf1", "y_acf5",
                                    "diff1y_acf1", "diff1y_acf5", "diff2y_acf1", "diff2y_acf5", "y_pacf5", "diff1y_pacf5", "diff2y_pacf5",
                                        "seas_acf1", "seas_pacf1", "firstmin_ac", "firstzero_ac", "holt_alpha", "holt_beta", "hw_alpha", "hw_beta", "hw_gamma"]
        # Relevant time-series features
        pca_all_working_factors = ["mean", "var", "entropy", "lumpiness", "stability", "hurst", "std1st_der", "crossing_points", "binarize_mean",
                                    "unitroot_kpss", "heterogeneity", "linearity", "trend_strength", "spikiness", "level_shift_idx",
                                    "level_shift_size", "y_acf1", "y_acf5", "diff1y_acf1", "diff1y_acf5", "diff2y_acf1", "diff2y_acf5",
                                    "y_pacf5", "diff1y_pacf5", "diff2y_pacf5", "seas_acf1", "seas_pacf1", "firstmin_ac", "firstzero_ac", "holt_alpha"]
        # Selected variables: redundant variables removed through visual inspection of PCA plots
        pca_all_selected_factors = ["entropy", "lumpiness", "stability", "hurst", "std1st_der", "crossing_points", "binarize_mean",
                                    "unitroot_kpss", "linearity", "trend_strength", "level_shift_idx", "y_acf1", "y_acf5", "diff1y_acf1", "diff1y_acf5",
                                    "y_pacf5", "diff1y_pacf5", "diff2y_pacf5", "seas_acf1", "seas_pacf1", "holt_alpha"]

        pca_nonuse_factors = [x for x in pca_all_avail_factors if x not in pca_all_selected_factors]

        # MAKE FEATURE: Annual time-series feature extraction (by taxa) for ANNUAL OUTLIER detection and for PCA use 
        year_vec = list(range(1970, 2016, 1))

        # Reextract date
        train = train.reset_index()

        # Define a feature that can extract a single taxa-specific time-series
        def extract_annual_ts(year_selection, taxa_selection):
            year_x = (train
                    .loc[:,["date",taxa_selection]]
                    .assign(year = lambda x: x["date"].dt.year)
                    .query("year == @year_selection")
                    .drop(columns="year")
            )
            return year_x

        # Make 2-tuples of all combinations of year and taxa
        crossing1 = [(x, y) for x in year_vec for y in taxa_list]

        # Run extract_annual_ts() on all tuples within crossing1 (use crossing1 as an index)
        yearly_ts = [extract_annual_ts(year, name) for year, name in crossing1]

        # Run the relevant kats module functions to extract time-series features from each time-series 
        def extract_ts_features(timeseries):
            ts = kats.consts.TimeSeriesData(timeseries, time_col_name = "date")
            ts_features_mod = kats.tsfeatures.tsfeatures.TsFeatures(selected_features = pca_all_selected_factors)
            xp_features = ts_features_mod.transform(ts)
            return xp_features

        yearly_ts_features = [extract_ts_features(ts) for ts in yearly_ts]
        len(yearly_ts_features)
        len(crossing1)

        # Parse results into an ordered DataFrame
        def pull_annual_taxa_features_dataframes(taxa_selection):
            # expr for val in colletion if condition
            # To get the indices: unpacks each tuple into 'year' and 'name'.
            indices = [index for index, (year, name) in enumerate(crossing1) if name == taxa_selection]
            # Use indices to retireve relevant list contents
            ts_features = [yearly_ts_features[x] for x in indices]
            # Create a dataframe from contents
            ts_features_pd = pd.DataFrame(ts_features)
            ts_features_pd["year"] = year_vec
            return(ts_features_pd)

        yearly_ts_features_bytaxa = [pull_annual_taxa_features_dataframes(taxa) for taxa in taxa_list]

        # Reduce dimensionality of the features with PCA, detect outliers using a Z score approach
        def outlier_finder_bytaxa(taxa):
            # Taxa indice
            taxa_ind = taxa_list.index(taxa)
            # Isolate taxa of interest
            x = yearly_ts_features_bytaxa[taxa_ind]
            # Some metrics may fail to converge = issues. Remove these columns
            x = x.dropna(axis=1)
            # Remove missing years ([0] due to nested nature)
            missing_data_years = np.where(x["entropy"]==0)[0]
            x = x.drop(missing_data_years, axis=0)
            # Set index
            x = x.set_index("year")
            # Pre-PCA scaling
            scaler = StandardScaler()
            x_std = scaler.fit_transform(x)
            # PCA
            pca = PCA()
            x_prcomp = pca.fit_transform(x_std)
            # Set scales
            cols_prcomp_range = range(x_prcomp.shape[1])
            cols_prcomp = []
            for pc in cols_prcomp_range:
                cols_prcomp.append(f"PC{(pc+1)}")
            # Tabulise results
            x_prcomp_pd = pd.DataFrame(x_prcomp, columns=cols_prcomp, index = x.index)
            # Apply zscore on the DataFrame
            df_zscore = x_prcomp_pd.apply(zscore)
            x_prcomp_pd["taxa"] = taxa
            # Set outlier threshold (picked) # higher threshold = less outliers
            threshold = 3.2
            # Identify outliers
            outliers = df_zscore[(np.abs(df_zscore) > threshold).any(axis=1)].copy()
            # Attach taxa name
            outliers["taxa"] = taxa
            return (outliers.loc[:,["taxa"]], x_prcomp_pd, pca)

        outliers_list = []
        outliers_full_pca_res = []
        pca_mods = []
        for taxa in taxa_list:
            res = outlier_finder_bytaxa(taxa)
            outliers_list.append(res[0])
            outliers_full_pca_res.append(res[1])
            pca_mods.append(res[2])

        outliers = pd.concat(outliers_list)
        outliers_full_pca_res = pd.concat(outliers_full_pca_res)
        pca_mods = list(zip(pca_mods, taxa_list))

        # Export
        with self.output().open("wb") as f:
            pickle.dump({'outliers': outliers,
                        'outliers_full_pca_res': outliers_full_pca_res,
                        "pca_mods": pca_mods,
                        "pca_all_selected_factors": pca_all_selected_factors}, f)

