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
    print(mae)
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")
    return mae.mean(), mae.std()


def cross_evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(X)

    mae_mean, mae_std = cross_evaluate(pipeline, X, y)
    text_score = f"{city} - MAE : {mae_mean:.3f} +/- {mae_std:.3f}"

    # plot_last_nth_results(pipeline, X, y, city)

    return text_score, mae_mean, mae_std


def get_save_params(
    score_sj, score_iq, mae_mean_sj, mae_std_sj, mae_mean_iq, mae_std_iq
):

    # Calculate scores
    Xsj, y = get_train_data("sj")
    Xiq, y = get_train_data("iq")
    isj, iiq, itot = len(Xsj), len(Xiq), len(Xsj) + len(Xiq)
    weighted_mean = mae_mean_sj * isj / itot + mae_mean_iq * iiq / itot
    weighted_std = mae_mean_sj * isj / itot + mae_mean_iq * iiq / itot

    # Get pipeline params and clean them
    pipeline = best_pipeline_intown(Xsj)
    data = str(pipeline.named_steps)
    data = data.replace("\n", "").replace("  ", "")
    data = (
        f"weighted average MAE: {weighted_mean:.3f} +/- {weighted_std:.3f}, "
        + str(score_sj)
        + ", "
        + str(score_iq)
        + ", "
        + data
    )

    # save params and score in file
    with open("results.txt", "a") as myfile:
        myfile.write(f"\n{data}")


if __name__ == "__main__":
    score_sj, mae_mean_sj, mae_std_sj = cross_evaluate_for_city("sj")
    score_iq, mae_mean_iq, mae_std_iq = cross_evaluate_for_city("iq")
    get_save_params(
        score_sj, score_iq, mae_mean_sj, mae_std_sj, mae_mean_iq, mae_std_iq
    )
