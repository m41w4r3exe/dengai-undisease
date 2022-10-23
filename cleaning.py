import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_test_data(city):
    official_testX = pd.read_csv("data/dengue_features_test.csv")
    city_filtered_testX = official_testX[official_testX.city == city]
    trimmed_testX = city_filtered_testX.drop(
        ["week_start_date", "city", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"], axis=1
    )
    return trimmed_testX


def get_train_data(city):
    official_trainX = pd.read_csv("./data/dengue_features_train.csv")
    official_trainY = pd.read_csv("./data/dengue_labels_train.csv")

    too_much_nas = official_trainX.isnull().sum(axis=1) < 4
    official_trainX = official_trainX[too_much_nas]
    official_trainY = official_trainY[too_much_nas]

    city_filtered_trainX = official_trainX[official_trainX.city == city]
    city_filtered_trainY = official_trainY[official_trainY.city == city]

    trimmed_trainX = city_filtered_trainX.drop(
        ["year", "week_start_date", "city", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"],
        axis=1,
    )
    trimmed_trainY = city_filtered_trainY.drop(["year", "weekofyear", "city"], axis=1)

    mean_of_target_y = trimmed_trainY.total_cases.mean()
    no_peaks = trimmed_trainY.total_cases < (mean_of_target_y * 6)

    peakless_trainX = trimmed_trainX[no_peaks]
    peakless_trainY = trimmed_trainY[no_peaks]

    return peakless_trainX, peakless_trainY


def  get_time_series_splitter(X):
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
