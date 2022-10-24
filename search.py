from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)
from cleaning import get_test_data, get_train_data, get_time_series_splitter
from best_pipeline import best_pipeline_intown
import numpy as np


def forest_grid_search_for_city(city):
    # Set gs parameters
    # params = dict(model__n_estimators=[100, 300], model__max_depth=[5, 10])
    params = dict(
        model__n_estimators=[int(x) for x in np.linspace(50, 20000, num=100)],
        model__max_depth=[int(x) for x in np.linspace(1, 100, num=100)],
        model__min_samples_split=np.linspace(0.01, 0.5, num=100),
        model__min_samples_leaf=np.linspace(0.01, 0.5, num=100),
        model__max_features=["auto", "sqrt", "log2"],
        model__bootstrap=[True, False],
        model__criterion=["absolute_error"],
        model__min_weight_fraction_leaf=np.linspace(0.01, 0.5, num=100),
        model__max_leaf_nodes=[int(x) for x in np.linspace(2, 200, num=100)],
        model__min_impurity_decrease=np.linspace(0, 0.5, num=100),
        model__ccp_alpha=np.linspace(0, 0.5, num=100),
    )

    # Run grid search and return gridsearch object
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(X)
    ts_cv = get_time_series_splitter(X)
    # gs = GridSearchCV(
    #     pipeline, params, scoring="neg_mean_absolute_error", n_jobs=-1, cv=ts_cv
    # )
    gs = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        cv=ts_cv,
        verbose=1,
        factor=2,
        min_resources=52,
    )
    gs.fit(X, y)
    return gs


def gradient_boost_grid_search_for_city(city):
    # Set gs parameters
    params = dict(
        model__loss=["absolute_error"],
        model__learning_rate=np.linspace(0.01, 0.5, num=10),
        model__n_estimators=[1000],
        # model__subsample=np.linspace(0, 1.0, num=100),
        model__criterion=['friedman_mse', 'squared_error', 'mse'],
        model__min_samples_split=np.linspace(0.01, 0.5, num=10),
        model__min_samples_leaf=np.linspace(0.01, 0.5, num=10),
        # model__min_weight_fraction_leaf=np.linspace(0.01, 0.5, num=100),
        model__max_depth=[int(x) for x in np.linspace(1, 10, num=10)],
        # model__min_impurity_decrease=np.linspace(0, 0.5, num=100),
        # model__model__max_features=["auto", "sqrt", "log2"],
        # model__max_leaf_nodes=[int(x) for x in np.linspace(2, 500, num=100)],
        # model__validation_fraction=np.linspace(0, 0.5, num=100),
        # model__tol=np.linspace(0.00001, 0.0005, num=100),
        # model__ccp_alpha=np.linspace(0, 0.5, num=100),
    )

    # Run grid search and return gridsearch object
    X, y = get_train_data(city)
    pipeline = best_pipeline_intown(X)
    ts_cv = get_time_series_splitter(X)

    gs = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        cv=ts_cv,
        verbose=1,
    )
    gs.fit(X, y)
    return gs


if __name__ == "__main__":
    # Model parameter search
    for city in ["sj", "iq"]:
        gs = gradient_boost_grid_search_for_city(city)
        print(city, "best params: ", gs.best_params_, "best score: ", -gs.best_score_)
