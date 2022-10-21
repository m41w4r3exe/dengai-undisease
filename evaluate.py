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

    return f' MAE : {mae.mean():.3f} +/- {mae.std():.3f}'


def evaluate_for_city(city):
    print(f"\nResults for {city.upper()}:")
    X, y = getTrainData(city)
    score = evaluate(best_pipeline_intown(X), X, y)

    return f'{city} {score}'

def get_save_params(score_sj, score_iq):
    X, y = getTrainData('sj')
    pipeline = best_pipeline_intown(X)
    data = str(pipeline.named_steps)
    data = data.replace('\n', '').replace('  ', '')
    data = score_sj + ', ' + score_iq + ', ' + data

    with open("results.txt", "a") as myfile:
        myfile.write(f'\n{data}')


if __name__ == "__main__":

    score_sj = evaluate_for_city("sj")
    score_iq = evaluate_for_city("iq")
    get_save_params(score_sj, score_iq)
