import pandas as pd


def getTrainData():
    offical_trainX = pd.read_csv("./data/dengue_features_train.csv")
    offical_trainY = pd.read_csv("./data/dengue_labels_train.csv")

    trimmed_trainX = offical_trainX.drop(["week_start_date"], axis=1)
    trimmed_trainY = offical_trainY.drop(["year", "weekofyear"], axis=1)

    ###############
    # Below section should be deleted after cross validation
    sj_official_trainX = trimmed_trainX[trimmed_trainX.city == "sj"]
    iq_official_trainX = trimmed_trainX[trimmed_trainX.city == "iq"]
    sj_trainX = sj_official_trainX[:749]
    iq_trainX = iq_official_trainX[:416]
    sj_testX = sj_official_trainX[749:]
    iq_testX = iq_official_trainX[416:]

    sj_official_trainY = trimmed_trainY[trimmed_trainY.city == "sj"]
    iq_official_trainY = trimmed_trainY[trimmed_trainY.city == "iq"]
    sj_trainY = sj_official_trainY[:749]
    iq_trainY = iq_official_trainY[:416]
    sj_testY = sj_official_trainY[749:]
    iq_testY = iq_official_trainY[416:]

    trainX = pd.concat((sj_trainX, iq_trainX), axis=0)
    trainY = pd.concat((sj_trainY, iq_trainY), axis=0)
    testX = pd.concat((sj_testX, iq_testX), axis=0)
    testY = pd.concat((sj_testY, iq_testY), axis=0)
    #############

    trainY = trainY.drop("city", axis=1)
    testY = testY.drop("city", axis=1)

    return trainX, trainY, testX, testY
