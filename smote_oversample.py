import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def add_norm_cases_and_peak_to_df(
    X, 
    Y,
    ):
    
    X['total_cases_normalized']=Y['total_cases']
    cities = X['city'].unique()
    for c in cities:
        ## converting numbers / ratio with inhabitants
        maxhere=Y['total_cases'].loc[Y['city']==c].max()
        indices = Y['total_cases'].loc[Y['city']==c].index
        for i in indices:
            X.loc[i, 'total_cases_normalized'] = (Y.loc[i, 'total_cases'])/maxhere

        # plotr to check the transformation
        # plt.plot(Y['total_cases'].loc[Y['city']==c])
        plt.show()
        plt.plot(X.loc[indices, 'total_cases_normalized'])
        plt.show()

    #adding new column in dataframe
    X['peak']=X['total_cases_normalized']
    X.loc[(X['total_cases_normalized']>=0.1, 'peak')]=1
    X.loc[(X['total_cases_normalized']<0.1, 'peak')]=0
    print(X['peak'].value_counts())

    return X


##
official_trainX = pd.read_csv("./data/dengue_features_train.csv")
official_trainY = pd.read_csv("./data/dengue_labels_train.csv")
official_trainX = add_norm_cases_and_peak_to_df(official_trainX, official_trainY)
official_trainX.head()




