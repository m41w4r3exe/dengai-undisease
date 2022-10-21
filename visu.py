import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# This is how to use this
# vizualize_fit(
#     trainY,
#     y_train_pred,
#     y_pred,
#     test_y=testY,
#     plot_y_test=True,
#     label='City_Model_Version',
#     mycolor = 'purple'


def vizualize_fit(
    train_y: pd.DataFrame,
    predict_train: np.array,
    predict_test: np.array,
    test_y: pd.DataFrame = None,
    plot_y_test: bool = False,
    label="City_Model_Version",
    mycolor="red",
):
    # defined targets
    y_train = train_y.loc[:, "total_cases"]
    y_test = test_y.loc[:, "total_cases"]

    cities = ["sj", "iq"]
    cities_colors = ["blue", "purple"]
    # train
    # city_data_log_train = trainY.loc[:,'city']==cities[ct]
    year_train = train_y.loc[:, "year"]
    week_train = train_y.loc[:, "weekofyear"]
    # print(zip(np.array(year), np.array(week)))
    time_train = [
        f"y{y}w{w}" for y, w in zip(np.array(year_train), np.array(week_train))
    ]

    # test
    # city_data_log = testY.loc[:,'city']==cities[ct]
    year_test = test_y.loc[:, "year"]
    week_test = test_y.loc[:, "weekofyear"]
    # print(zip(np.array(year), np.array(week)))
    time_test = [f"y{y}w{w}" for y, w in zip(np.array(year_test), np.array(week_test))]
    # print(f'TRAIN - MAE =  {mean_absolute_error(Y_train, predict_train)}')
    # print(f'TEST - MAE =  {mean_absolute_error(Y_test, predict_test)}')

    train_mae = mean_absolute_error(y_train, predict_train)
    test_mae = mean_absolute_error(y_test, predict_test)

    fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

    real_vs_predict(
        y_train,
        predict_train,
        data_type="Train",
        mae=train_mae,
        ax=axs[0],
        mycolor=mycolor,
    )
    real_vs_predict(
        y_test, predict_test, data_type="Test", mae=test_mae, ax=axs[1], mycolor=mycolor
    )
    plt.tight_layout()
    fig.savefig(f"Correlation_Real_Prediction_{label}")

    fig, axs = plt.subplots(figsize=(10, 12), nrows=2)
    # ploting real and predicted as a fucntio of time - train set
    compare_real_pred(
        y_train, predict_train, axs[0], time_train, color=mycolor, label="Training Set"
    )
    # ploting real and predicted as a fucntio of time - test set
    compare_real_pred(
        y_test, predict_test, axs[1], time_test, color=mycolor, label="Test Set"
    )
    plt.tight_layout()
    fig.savefig(f"Comparison_Real_Prediction_{label}")


##
def real_vs_predict(
    y_true: pd.Series,
    y_pred: np.array,
    data_type: str,
    mae: float,
    ax: plt.axes,
    mycolor="purple",
):
    ax.scatter(y_true, y_pred, marker="o", color=mycolor)
    max_value = max(max(y_true), max(y_pred))
    min_value = min(min(y_true), min(y_pred))
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    line = [[min_value, min_value], [max_value, max_value]]

    ax.plot(line, line, linestyle="--", color="red")
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{data_type} Real")
    ax.set_ylabel(f"{data_type} Prediction")


##
def compare_real_pred(y_true, y_pred, ax, x_val=[], color="black", label=""):

    ax.scatter(x_val, y_true, marker="o", color="grey")
    ax.plot(x_val, y_pred, color=color)
    ax.set_ylabel("Total Cases")
    ax.set_xlabel("Time")
    ax.set_title(label)
    x_ti = [x_val[l] for l in range(0, len(x_val), 20)]
    ax.set_xticks(x_ti)
    ax.set_xticklabels(x_ti, rotation="vertical")
    ax.autoscale(enable=True, axis="both", tight=True)
