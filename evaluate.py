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
        scoring=["neg_mean_absolute_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n")

    return mae.mean(), mae.std()


def evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = getTrainData(city)
    mae_mean, mae_std = evaluate(best_pipeline_intown(X), X, y)
    text_score= f'{city} - MAE : {mae_mean:.3f} +/- {mae_std:.3f}'

    return text_score, mae_mean, mae_std

def get_save_params(score_sj, score_iq, mae_mean_sj, mae_std_sj, mae_mean_iq, mae_std_iq):

    # Calculate scores
    Xsj, y = getTrainData('sj')
    Xiq, y = getTrainData('iq')
    isj, iiq, itot = len(Xsj), len(Xiq), len(Xsj) + len(Xiq)
    weighted_mean = mae_mean_sj * isj / itot + mae_mean_iq * iiq / itot
    weighted_std = mae_mean_sj * isj / itot + mae_mean_iq * iiq / itot

    # Get pipeline params and clean them
    pipeline = best_pipeline_intown(Xsj)
    data = str(pipeline.named_steps)
    data = data.replace('\n', '').replace('  ', '')
    data = f'weighted average MAE: {weighted_mean:.3f} +/- {weighted_std:.3f}, '  + str(score_sj) + ', ' + str(score_iq) + ', ' + data

    # save params and score in file
    with open("results.txt", "a") as myfile:
        myfile.write(f'\n{data}')


if __name__ == "__main__":

    score_sj, mae_mean_sj, mae_std_sj = evaluate_for_city("sj")
    score_iq, mae_mean_iq, mae_std_iq = evaluate_for_city("iq")
    get_save_params(score_sj, score_iq, mae_mean_sj, mae_std_sj, mae_mean_iq, mae_std_iq)
