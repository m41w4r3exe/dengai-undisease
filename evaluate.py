from best_pipeline import best_pipeline_intown
from sklearn.model_selection import cross_validate
from cleaning import (
    get_time_series_splitter,
    get_train_data,
)
from visu import plot_last_nth_results


def cross_evaluate(pipeline, X, y):
    ts_cv = get_time_series_splitter(X)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=ts_cv,
        scoring=["neg_mean_absolute_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")


def cross_evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(X)

    plot_last_nth_results(pipeline, X, y, city)

    cross_evaluate(pipeline, X, y)


cross_evaluate_for_city("sj")
cross_evaluate_for_city("iq")
