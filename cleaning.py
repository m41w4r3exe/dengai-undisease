import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_test_data(city):
    official_testX = pd.read_csv("data/dengue_features_test.csv")
    city_filtered_trainX = official_testX[official_testX.city == city]
    trimmed_testX = city_filtered_trainX.drop(["week_start_date", "city"], axis=1)
    return trimmed_testX


def get_train_data(city):
    official_trainX = pd.read_csv("./data/dengue_features_train.csv")
    official_trainY = pd.read_csv("./data/dengue_labels_train.csv")

    city_filtered_trainX = official_trainX[official_trainX.city == city]
    city_filtered_trainY = official_trainY[official_trainY.city == city]

    trimmed_trainX = city_filtered_trainX.drop(["week_start_date", "city"], axis=1)
    trimmed_trainY = city_filtered_trainY.drop(["year", "weekofyear", "city"], axis=1)

    # ndvi
    ndvi_cols = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
    trimmed_trainX.drop(ndvi_cols, axis=1, inplace=True)

    # Precipitation
    precipitation_cols = [
        "precipitation_amt_mm",
        "reanalysis_precip_amt_kg_per_m2",
        "reanalysis_sat_precip_amt_mm",
        "station_precip_mm",
    ]
    # trimmed_trainX["precip_avg"] = trimmed_trainX[precipitation_cols].mean(axis=1)
    trimmed_trainX.drop(precipitation_cols, axis=1, inplace=True)

    # temperature
    temp_cols = [
        "reanalysis_air_temp_k",
        "reanalysis_avg_temp_k",
        "station_avg_temp_c",
        "reanalysis_tdtr_k",
    ]
    trimmed_trainX["temp_avg"] = trimmed_trainX[temp_cols].mean(axis=1)
    trimmed_trainX.drop(temp_cols, axis=1, inplace=True)

    # Max temperature
    max_temp = ["reanalysis_max_air_temp_k", "station_max_temp_c"]
    trimmed_trainX["max_temp_avg"] = trimmed_trainX[max_temp].mean(axis=1)
    trimmed_trainX.drop(max_temp, axis=1, inplace=True)

    # Min temperature
    min_temp = ["reanalysis_min_air_temp_k", "station_min_temp_c"]
    trimmed_trainX["min_temp_avg"] = trimmed_trainX[min_temp].mean(axis=1)
    trimmed_trainX.drop(min_temp, axis=1, inplace=True)

    # Temperature range
    temp_range = ["station_diur_temp_rng_c", "temp_range_avg"]
    trimmed_trainX["temp_range_avg"] = (
        trimmed_trainX.max_temp_avg - trimmed_trainX.min_temp_avg
    )
    trimmed_trainX["temp_range_avg"] = trimmed_trainX[temp_range].mean(axis=1)
    trimmed_trainX.drop("station_diur_temp_rng_c", axis=1, inplace=True)

    # Humidity
    humidity_cols = [
        "reanalysis_relative_humidity_percent",
        "reanalysis_specific_humidity_g_per_kg",
    ]
    # trimmed_trainX["humidity_avg"] = trimmed_trainX[humidity_cols].mean(axis=1)
    trimmed_trainX.drop(humidity_cols, axis=1, inplace=True)

    return trimmed_trainX, trimmed_trainY


def get_time_series_splitter(X):
    row = X.shape[0]
    n_splits = round(row / 52)
    return TimeSeriesSplit(n_splits=n_splits, gap=0)


def split_nth_last_part(X, y, n=0):
    """Splits data set into train and test by splitting it into n and returns the last splitted sets
    if n is not given, it takes it from TimeSeriesSplitter"""
    nth_split = n
    if nth_split == 0:
        nth_split = get_time_series_splitter(X).get_n_splits()
    row = X.shape[0]
    splitting_index = round(row * ((nth_split - 1) / nth_split))

    trainX = X[:splitting_index]
    testX = X[splitting_index:]

    trainY = y[:splitting_index]
    testY = y[splitting_index:]

    return trainX, trainY, testX, testY
