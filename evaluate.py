from best_pipeline import best_pipeline_intown
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from cleaning import getTrainData


def evaluate(pipeline, X, y):
    ts_cv = TimeSeriesSplit(n_splits=10, gap=5)

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=ts_cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")


def evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = getTrainData(city)
    evaluate(best_pipeline_intown(X), X, y)


evaluate_for_city("sj")
evaluate_for_city("iq")
