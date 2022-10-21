import pandas as pd


def getTrainData(city: str = ""):
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
