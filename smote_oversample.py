import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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


    def oversample_smote_X(join_X_Y, on='peak'):
        resampling_col = join_X_Y.loc[:, on]
        dropped_join_X_Y = join_X_Y.drop(columns=["peak", "city", "week_start_date"])
        # dropped_join_X_Y = join_X_Y.drop(columns=["peak")

        pipe = make_pipeline(
            SimpleImputer(),
            SMOTE(sampling_strategy=0.5, random_state=42),
        )
        upsampled_X, upsampled_y = pipe.fit_resample(X, resampling_col)
        return upsampled_X



 

##
official_trainX = pd.read_csv("./data/dengue_features_train.csv")
official_trainY = pd.read_csv("./data/dengue_labels_train.csv")
official_trainX = add_norm_cases_and_peak_to_df(official_trainX, official_trainY)


official_trainX['peak'].value_counts()/


oversample_smote_X_Y(official_trainX)