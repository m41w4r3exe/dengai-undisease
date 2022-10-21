from best_pipeline import best_pipeline_intown
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate

# total_abs_error = 0
# runtime_amount = 30

# def find_mae(trainX, trainY, testX, testY):
#     for i in range(runtime_amount):
#         predicted_Y = best_pipeline_intown(trainX, trainY, testX)

#     absolute_error = mean_absolute_error(testY, predicted_Y)
#     total_abs_error += absolute_error
#     print(i + 1, "MAE:", absolute_error)


# print("Average MAE:", total_abs_error / runtime_amount)


def evaluate(pipeline, X, y, cv):
    ts_cv = TimeSeriesSplit(n_splits=64, gap=0)
    y = y.total_cases
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=ts_cv,
        scoring=["pos_mean_absolute_error"],
    )
    mae = -cv_results["test_pos_mean_absolute_error"]
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")
