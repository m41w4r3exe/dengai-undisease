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
        pipeline, X, y, cv=ts_cv, scoring=["neg_mean_absolute_error"], n_jobs=-1
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    print(mae)
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")
    return mae


def cross_evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(X)
    mae = cross_evaluate(pipeline, X, y)
    plot_last_nth_results(pipeline, X, y, city)

    return mae


def get_save_params(mae_sj, mae_iq):

    # Calculate scores
    Xsj, _ = get_train_data("sj")
    Xiq, _ = get_train_data("iq")
    isj, iiq, itot = len(Xsj), len(Xiq), len(Xsj) + len(Xiq)
    weighted_mean = mae_sj.mean() * isj / itot + mae_iq.mean() * iiq / itot
    weighted_std = mae_sj.std() * isj / itot + mae_iq.std() * iiq / itot

    # prepare strings for scores
    score_sj = f"MAE sj: {mae_sj.mean():.3f} +/- {mae_sj.std():.3f}"
    score_iq = f"MAE iq: {mae_iq.mean():.3f} +/- {mae_iq.std():.3f}"

    # Get pipeline params and clean them
    pipeline = best_pipeline_intown(Xsj)
    data = str(pipeline.named_steps["model"])
    data = data.replace("\n", "").replace("  ", "")
    data = (
        f"weighted average MAE: {weighted_mean:.3f} +/- {weighted_std:.3f}, "
        + str(score_sj)
        + ", "
        + str(score_iq)
        + ", "
        + data
    )

    print(data)

    # save params and score in file
    with open("results.txt", "a") as myfile:
        myfile.write(f"\n{data}")


if __name__ == "__main__":
    # Model cross-evaluation
    mae_sj = cross_evaluate_for_city("sj")
    mae_iq = cross_evaluate_for_city("iq")
    get_save_params(mae_sj, mae_iq)
