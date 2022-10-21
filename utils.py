import pandas as pd
from datetime import datetime
from best_pipeline import best_pipeline_intown
from cleaning import get_test_data, get_train_data


def export_results(city):
    testX_sj = get_test_data(city)
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(testX_sj)
    pipeline.fit(X, y)
    predicted_Y = pipeline.predict(testX_sj)
    rounded_y = testX_sj.iloc[:, :2]
    rounded_y.insert(0, "city", city)
    rounded_y["total_cases"] = predicted_Y.round(0).astype(int)
    return rounded_y


def export_csv():
    sj_results = export_results("sj")
    iq_results = export_results("iq")

    results = pd.concat([sj_results, iq_results])

    results.to_csv(f"./results/dengue_preds_{get_current_time()}.csv", index=False)


def get_current_time():
    currentDateAndTime = datetime.now()
    return currentDateAndTime.strftime("%H_%M_%S")


if __name__ == "__main__":
    export_csv()
