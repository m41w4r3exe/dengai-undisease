import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_train_data(city: str = ""):
    official_trainX = pd.read_csv("./data/dengue_features_train.csv")
    official_trainY = pd.read_csv("./data/dengue_labels_train.csv")

    trimmed_trainX = official_trainX.drop(["week_start_date"], axis=1)
    trimmed_trainY = official_trainY.drop(["year", "weekofyear"], axis=1)

    if city != "":
        city_filtered_trainX = trimmed_trainX[trimmed_trainX.city == city]
        city_filtered_trainY = trimmed_trainY[trimmed_trainX.city == city]
        city_filtered_trainY = city_filtered_trainY.drop("city", axis=1)
        return city_filtered_trainX, city_filtered_trainY

    trimmed_trainY = trimmed_trainY.drop("city", axis=1)

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
