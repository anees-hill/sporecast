import pandas as pd
import numpy as np

# Discretise wind direction degrees
def reclassify_direction(direction):
    if (direction >= 348.75) or (direction < 11.25):
        return 1  # N
    elif (direction >= 11.25) and (direction < 33.75):
        return 2  # NNE
    elif (direction >= 33.75) and (direction < 56.25):
        return 3  # NE
    elif (direction >= 56.25) and (direction < 78.75):
        return 4  # ENE
    elif (direction >= 78.75) and (direction < 101.25):
        return 5  # E
    elif (direction >= 101.25) and (direction < 123.75):
        return 6  # ESE
    elif (direction >= 123.75) and (direction < 146.25):
        return 7  # SE
    elif (direction >= 146.25) and (direction < 168.75):
        return 8  # SSE
    elif (direction >= 168.75) and (direction < 191.25):
        return 9  # S
    elif (direction >= 191.25) and (direction < 213.75):
        return 10  # SSW
    elif (direction >= 213.75) and (direction < 236.25):
        return 11  # SW
    elif (direction >= 236.25) and (direction < 258.75):
        return 12  # WSW
    elif (direction >= 258.75) and (direction < 281.25):
        return 13  # W
    elif (direction >= 281.25) and (direction < 303.75):
        return 14  # WNW
    elif (direction >= 303.75) and (direction < 326.25):
        return 15  # NW
    elif (direction >= 326.25) and (direction < 348.75):
        return 16  # NNW
    else:
        return None

# Helper function for applying reclassify_direction() in a DataFrame
def add_direction_class(df, column_name, new_column_name):
    df[new_column_name] = df[column_name].apply(reclassify_direction)

# Calculate the days since a meteorological event occured. Utilises a reference table containing
#  defintions of what a 'meteorological event' is for each fungal taxa.
def add_days_since_meteorological_event(df, taxa, threshold_reference_df):
    # Define a counting (nested) function for use in this function
    def reset_cumulative_sum(series, reset_value):
        cum_sum = 0
        result = []
        for x in series:
            cum_sum += x
            if cum_sum >= reset_value:
                cum_sum = 0
            result.append(cum_sum)
        return pd.Series(result, index=series.index)

    # Import taxa-specific thresholds
    threshold_reference_x = threshold_reference_df[threshold_reference_df["taxa"]==taxa].copy()
    temp_threshold_x = threshold_reference_x.loc[:,"temp_threshold"].item()
    precip_threshold_x = threshold_reference_x.loc[:,"precip_threshold"].item()
    max_wind_speed_threshold_x = threshold_reference_x.loc[:,"max_wind_speed_threshold"].item()
    
    # Copy
    data = df.copy()
    data = data.reset_index()

    # TEMP MAX
    data['indicator'] = np.where(data['c_max_air_temp'] >= temp_threshold_x, 1, 100000)
    data['cum_sum'] = reset_cumulative_sum(data['indicator'], 1000)
    data['days_since_threshold_temp_max'] = np.where(data['cum_sum'] > 1000, 0, data['cum_sum'])
    data['days_since_threshold_temp_max_lag'] = data['days_since_threshold_temp_max'].shift()
    
    # TEMP MIN
    data['indicator'] = np.where(data['c_min_air_temp'] >= temp_threshold_x, 1, 100000)
    data['cum_sum'] = reset_cumulative_sum(data['indicator'], 1000)
    data['days_since_threshold_temp_min'] = np.where(data['cum_sum'] > 1000, 0, data['cum_sum'])
    data['days_since_threshold_temp_min_lag'] = data['days_since_threshold_temp_min'].shift()
    
    # PRECIPITATION
    data['indicator'] = np.where(data['c_prcp_amt'] >= precip_threshold_x, 1, 100000)
    data['cum_sum'] = reset_cumulative_sum(data['indicator'], 1000)
    data['days_since_threshold_precip'] = np.where(data['cum_sum'] > 1000, 0, data['cum_sum'])
    data['days_since_threshold_precip_lag'] = data['days_since_threshold_precip'].shift()
    
    # WIND SPEED
    data['indicator'] = np.where(data['c_max_mean_wind_speed'] >= max_wind_speed_threshold_x, 1, 100000)
    data['cum_sum'] = reset_cumulative_sum(data['indicator'], 1000)
    data['days_since_threshold_max_mean_wind_speed'] = np.where(data['cum_sum'] > 1000, 0, data['cum_sum'])
    data['days_since_threshold_gustspeed_max_lag'] = data['days_since_threshold_max_mean_wind_speed'].shift()
    
    # Drop unnecessary columns
    data = data.drop(columns=['indicator', 'cum_sum', 'days_since_threshold_max_mean_wind_speed', 'days_since_threshold_precip',
                          'days_since_threshold_temp_min', 'days_since_threshold_temp_max'])
    
    return data

# Assign an intiger value for each observation to indicate the associated sampling site
def assign_site(df):
    df_x = df.copy()
    df_x = df_x.reset_index()
    df_x['date'] = pd.to_datetime(df_x['date'])
    # Assign site integer
    df_x['site'] = np.where(df_x['date'] < '1983-01-01', 0, 
                          np.where((df_x['date'] >= '1983-01-01') & (df_x['date'] < '1990-01-01'), 1, 
                                   np.where((df_x['date'] >= '1990-01-01') & (df_x['date'] < '2006-01-01'), 2, 3)))
    df_x = df_x.set_index("date")
    return df_x

# Assign an integer based on the decade
def assign_decade_cat(df):
    df_x = df.copy()
    df_x = df_x.reset_index()
    df_x['date'] = pd.to_datetime(df_x['date'])
    # Assign decade integer
    df_x['decade_cat'] = np.where(df_x['date'] < '1980-01-01', 0, 
                                np.where((df_x['date'] >= '1980-01-01') & (df_x['date'] < '1990-01-01'), 1,
                                         np.where((df_x['date'] >= '1990-01-01') & (df_x['date'] < '2000-01-01'), 2,
                                                  np.where((df_x['date'] >= '2000-01-01') & (df_x['date'] < '2010-01-01'), 3, 
                                                           np.where(df_x['date'] >= '2010-01-01', 4, np.nan)))))
    df_x = df_x.set_index("date")
    return df_x

# Extracts temporal features from the 'date' column
def add_temporal_features(df):
    df_x = df.copy()
    df_x = df_x.reset_index()
    df_x['date'] = pd.to_datetime(df_x['date'])
    # Extract year, month and day of year
    df_x['year'] = df_x['date'].dt.year
    df_x['month'] = df_x['date'].dt.month
    df_x['DoY'] = df_x['date'].dt.dayofyear
    # Add dates
    for i in range(2, 8):
        df_x[f'date_{i}'] = df_x['date'] + pd.Timedelta(days=i-1)
    # Weeks (using Monday ('_1') as the first day of the week)
    df_x['week_1'] = df_x['date'].dt.isocalendar().week
    for i in range(2, 8):
        df_x[f'week_{i}'] = df_x[f'date_{i}'].dt.isocalendar().week
    # Drop unnecessary columns
    cols_to_drop = [f'date_{i}' for i in range(2, 8)] + [f'week_{i}' for i in range(2, 8)]
    df_x = df_x.drop(columns=cols_to_drop)
    # Convert week_1 to float
    df_x['week_1'] = df_x['week_1'].astype(float)
    df_x = df_x.set_index("date")
    return df_x

# Calculate `growing degree days` using a reference table `meteo_thresholds` for the fungal taxa of interest
def growing_degree_days_group(df, var, meteo_thresholds, max_temp_col='c_max_air_temp', min_temp_col='c_min_air_temp'):
    data = df.copy()
    # Find base_temp from `meteo_thresholds` GAM model output
    meteo_thresholds_matching = meteo_thresholds.loc[meteo_thresholds["taxa"] == var, "temp_threshold"]
    if len(meteo_thresholds_matching) == 0:
        print(f"No temperature threshold found for variable: {var}")
        return data
    meteo_threshold = meteo_thresholds_matching.values[0]
    # Calculate `gdd`
    data['c_gdd'] = (data[max_temp_col] + data[min_temp_col]) / 2
    data['c_gdd'] = data['c_gdd'].apply(lambda row: round(max(0, row - meteo_threshold), 2))
    return data

# Select the annual period to model. Makes use of reference table with calculated seasonal dates
def trim_seasons(df, var, seasonal_dates):
    data = df.copy()
    # locate relevant seasonal dates
    seasonal_dates_matching = seasonal_dates.loc[seasonal_dates["taxa"]==var,["season_start_earliest", "season_end_latest"]]
    if len(seasonal_dates_matching) == 0:
        print(f"No seasonal date details found for variable: {var}")
        return data
    start_date = seasonal_dates_matching.iloc[0,0]
    end_date =  seasonal_dates_matching.iloc[0,1]
    # Trim outer-season dates from data.frame
    dat_trimmed = data.query("DoY >= @start_date & DoY <= @end_date")
    return dat_trimmed

# For each fungal taxa, identify 'peaks' based on the following criteria:
def peak_classify(df, variable_selection, peak_thresholds_table_input, plot):
    # Include function to find peaks
    def findAeroPeaks(data, prop, threshold):
        df = pd.DataFrame({'value': data})
        df['value_lag1'] = df['value'].shift(1)
        df['value_lag2'] = df['value'].shift(2)
        df['value_lag3'] = df['value'].shift(3)
        df['prop_inc1'] = (df['value'] - df['value_lag1']) / df['value']
        df['prop_inc2'] = (df['value'] - df['value_lag2']) / df['value']
        df['prop_inc3'] = (df['value'] - df['value_lag3']) / df['value']
        df['over_thres'] = (data >= threshold).astype(int)
        df['peak'] = ((df['prop_inc1'] >= prop) & (df['over_thres'] == 1)).astype(int)
        df['ongoing_peak'] = ((df['peak'].shift(1) == 1) & (df['prop_inc1'] >= prop)).astype(int)
        df['peak'] = (df['ongoing_peak'] == 0) * df['peak']
        df['peak'] = df['peak'].fillna(0)
        return df['peak']
    # Utilise function on input dataset, per variable
    threshold_x = peak_thresholds_table_input.loc[peak_thresholds_table_input['variable'] == variable_selection, 'threshold'].values[0]
    # print(f"Processing peaks: {variable_selection}    - threshold: {threshold_x}")
    x = df.sort_values('date').copy()
    x.rename(columns={variable_selection: 'value'}, inplace=True)
    x['peak'] = findAeroPeaks(x['value'].values, prop=0.4, threshold=threshold_x)
    x['year'] = x['date'].dt.year
    x['variable'] = variable_selection
    years_vec = sorted(x['year'].unique())
    if plot:
        plot_output = {year: x[x['year'] == year].plot(x='date', y='value', kind='line', title=str(year), color=x['peak'].map({0: 'blue', 1: 'red'})) for year in years_vec}
    else:
        plot_output = {}
    x = x.loc[:,["date","peak"]]
    return {'data': x, 'plots': plot_output}


# Add 'peak' column according to definition in reference table `peak_reference_table`
def incorporate_peaks(df, var, peak_reference_table):
    data = df.copy()
    # locate relevant seasonal dates
    peak_ref_x = peak_reference_table[var]["data"]
    # Merge with data
    df_merged = pd.merge(df, peak_ref_x, on=["date"], how="left")
    return df_merged

# Incorporate the clusters calculated from the 'find_doy_clusters.py' script
def incorporate_clusters(df, var, cluster_reference_table):
    data = df.copy()
    # locate relevant seasonal dates
    cluster_ref_x = cluster_reference_table.loc[cluster_reference_table["taxa"]==var,["year", "DoY", "year_type_for_doy"]]
    # Merge with data
    df_merged = pd.merge(data, cluster_ref_x, on=["year", "DoY"], how="left")
    # Convert non-clustering period values to zero
    inds = df_merged["year_type_for_doy"].isnull()
    inds = inds[inds == True]
    df_merged.loc[inds.index, "year_type_for_doy"] = 0
    # Keep lag1 only (lag0 would result in data leakage)
    df_merged["year_type_for_doy_lag1"] = df_merged["year_type_for_doy"].shift(1)
    df_merged = df_merged.drop("year_type_for_doy", axis=1)
    return df_merged

# Generate lagged versions of specified columns
def generate_lagged_features(df, lag_columns, max_lags):
    for col_name in lag_columns:
        for lag in range(1, max_lags + 1):
            df[col_name + '_lag' + str(lag)] = df[col_name].shift(lag)
    return df

# Prepare the individual taxa-datasets for export (for modelling) by removing irrelvant categories
#  calculating the sum of other fungal spore counts, and general tidying of the data:
def merge_non_target_spores(group, name, original_data, value_vars, lag_model):
    # Get the date range in the group
    min_date, max_date = group['date'].min(), group['date'].max()

    # Select the relevant slice of the original data
    slice_df = original_data.loc[(original_data['date'] >= min_date) & (original_data['date'] <= max_date)].copy()

    # Create a copy of value_vars list excluding the current variable
    # Total_fungal_spores is not included as it includes detail from the outcome variable
    other_vars = [var for var in value_vars if var not in [name, "Total_fungal_spores"]]

    # Retain 'date' and the other_vars in slice_df
    slice_df = slice_df[['date'] + other_vars]

    # Add a new column with the sum of the other variables
    slice_df['sum_other_vars'] = slice_df[other_vars].sum(axis=1)

    # Here is the correction: use extend instead of append for adding multiple items to a list
    other_vars.extend(['date', 'sum_other_vars'])

    # Merge the slice with the group
    merged_df = pd.merge(group, slice_df[other_vars], on='date', how='left')

    # Rename '[name]' as 'outcome'
    merged_df = merged_df.rename({"value": "outcome"}, axis=1)
    merged_df = merged_df.drop("variable", axis=1)

    # Creating lags
    for col in other_vars + ['outcome']:
        for lag in range(1, lag_model+1):
            merged_df[f"{col}_lag{lag}"] = merged_df[col].shift(lag)

    return merged_df


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
