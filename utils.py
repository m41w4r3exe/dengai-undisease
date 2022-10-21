import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from datetime import datetime


def export_csv_results(pipeline):
    official_testX = pd.read_csv("data/dengue_features_test.csv")
    predsY = pipeline.predict(official_testX)
    preds_df = official_testX.iloc[:, :3]
    preds_df["total_cases"] = predsY.round(0).astype(int)
    preds_df.to_csv(f"./results/dengue_preds_{get_current_time()}.csv", index=False)


class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Put a breakpoint to below return line to debug transformed values
        # TODO:make plots of transformed and non transformed data
        return data

    def fit(self, data, y=None, **fit_params):

        return self


def get_current_time():
    currentDateAndTime = datetime.now()
    return currentDateAndTime.strftime("%H_%M_%S")
