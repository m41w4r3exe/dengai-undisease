# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


def data_and_target_df_import():
    X = pd.read_csv("./data/dengue_features_train.csv")
    Y = pd.read_csv("./data/dengue_labels_train.csv")
    return X, Y


def combine_X_and_Y_dataFrames(
    X: pd.DataFrame,
    Y: pd.DataFrame,
):
    combined_X_Y = X.copy()
    combined_X_Y["total_cases"] = Y["total_cases"]
    return combined_X_Y


def plot_peaks_on_time_series(
    combined_X_Y: pd.DataFrame,
    city: str = "city_name",
    color: str = "purple",
    peak_threshold: float = 0.1,
):
    # plot to check the transformation and show the peaks
    indices = combined_X_Y["total_cases"].loc[combined_X_Y["city"] == city].index
    date = combined_X_Y["week_start_date"].loc[combined_X_Y["city"] == city]
    plt.plot(
        date.loc[indices],
        combined_X_Y.loc[indices, "total_cases_normalized"],
        color=color,
    )
    tick_select = range(min(indices), max(indices), 52)
    plt.xticks(date.loc[tick_select], rotation="vertical")
    log_peak = combined_X_Y.loc[indices, "total_cases_normalized"] > peak_threshold
    plt.plot(
        date.loc[indices[log_peak]],
        combined_X_Y.loc[indices[log_peak], "total_cases_normalized"],
        "o",
        color="red",
    )
    plt.title(f"city: {city}", color=color)
    plt.xlabel("date")
    plt.ylabel("normalized case numbers")
    plt.axis("tight")
    plt.show()


def histo_norm_cases(combined_X_Y):
    plt.hist(combined_X_Y["total_cases_normalized"], 40, color="grey")
    plt.title("distribution of case numbers in dataset")
    plt.xlabel("normalized case numbers")
    plt.ylabel("count")
    plt.axis("tight")
    plt.show()
    pass


def add_norm_cases_and_peak_to_df(
    combined_X_Y: pd.DataFrame,
    city: str = "cityname",
    color: str = "purple",
    plot_peaks: bool = True,
):
    # Adding a column "normalized_case_numbers" and
    # Adding a column "peak" labelling the outbreaks of the epidemics with 1
    peak_threshold = 0.07
    combined_X_Y.insert(
        len(combined_X_Y.columns),
        "total_cases_normalized",
        combined_X_Y.loc[:, "total_cases"],
    )

    ## converting numbers / ratio with inhabitants
    max_cases = combined_X_Y["total_cases"].max()
    indices = combined_X_Y["total_cases"].index
    for i in indices:
        combined_X_Y.loc[i, "total_cases_normalized"] = (
            combined_X_Y.loc[i, "total_cases"]
        ) / max_cases

    if plot_peaks:
        plot_peaks_on_time_series(
            combined_X_Y, city=city, color=color, peak_threshold=peak_threshold
        )
        histo_norm_cases(combined_X_Y)

    # adding "peak" column to dataframe -> defined to categorize the target as epidemic peak (1) or not (0)
    combined_X_Y.insert(combined_X_Y.shape[1], "peak", np.zeros(combined_X_Y.shape[0]))
    # combined_X_Y["peak"] = combined_X_Y["total_cases_normalized"]
    combined_X_Y.loc[
        (combined_X_Y["total_cases_normalized"] >= peak_threshold, "peak")
    ] = 1
    combined_X_Y.loc[
        (combined_X_Y["total_cases_normalized"] < peak_threshold, "peak")
    ] = 0
    return combined_X_Y


def encode_cat_features(combined_X_Y):
    all_categorical_features = combined_X_Y.select_dtypes(include=[object]).columns
    categorical_features = [value for value in all_categorical_features]
    cat_encoder = OrdinalEncoder()
    combined_X_Y_pre = combined_X_Y.copy()
    combined_X_Y.loc[:, categorical_features] = cat_encoder.fit_transform(
        combined_X_Y[categorical_features]
    )
    return combined_X_Y, cat_encoder, categorical_features


def decode_cat_features(combined_X_Y, cat_encoder, categorical_features):
    combined_X_Y.loc[:, categorical_features] = cat_encoder.inverse_transform(
        combined_X_Y.loc[:, categorical_features]
    )
    return combined_X_Y


def smote_oversample_of_peaks(combined_X_Y, based_on="peak"):
    resampling_col = combined_X_Y.loc[:, based_on]
    combined_X_Y = combined_X_Y.drop(columns=[based_on])
    combined_X_Y = combined_X_Y.drop(columns="total_cases_normalized")
    pipe = make_pipeline(
        SimpleImputer(),
        SMOTE(sampling_strategy=0.75, random_state=42),
    )
    upsampled_X_Y, upsampled_col = pipe.fit_resample(combined_X_Y, resampling_col)
    upsampled_X_Y = pd.DataFrame(upsampled_X_Y, columns=combined_X_Y.columns)

    # reorder and reindex by date
    upsampled_X_Y = upsampled_X_Y.sort_values(by=["year", "weekofyear"])
    return upsampled_X_Y


def split_X_Y(upsampled_X_Y):
    from_x_to_y = ["city", "year", "weekofyear", "total_cases"]
    Y_upsampled = upsampled_X_Y[from_x_to_y]
    X_upsampled = upsampled_X_Y.drop(columns="total_cases")
    return X_upsampled, Y_upsampled


def smote_oversample_run_all():
    X, Y = data_and_target_df_import()
    combined_X_Y = combine_X_and_Y_dataFrames(X, Y)

    colors = ["purple", "orange"]
    cities = set(combined_X_Y["city"].unique())
    X_upsampled = pd.DataFrame()
    Y_upsampled = pd.DataFrame()
    for c, color in zip(cities, colors):
        this_city_combined_X_Y = pd.DataFrame()
        this_city_upsampled_X_Y = pd.DataFrame()
        this_city_X_upsampled = pd.DataFrame()
        this_city_Y_upsampled = pd.DataFrame()

        this_city_combined_X_Y = combined_X_Y.loc[combined_X_Y["city"] == c, :]
        this_city_combined_X_Y = add_norm_cases_and_peak_to_df(
            this_city_combined_X_Y, city=c, color=color
        )
        this_city_combined_X_Y, cat_encoder, categorical_features = encode_cat_features(
            this_city_combined_X_Y
        )
        this_city_upsampled_X_Y = smote_oversample_of_peaks(
            this_city_combined_X_Y, based_on="peak"
        )
        this_city_upsampled_X_Y = decode_cat_features(
            this_city_upsampled_X_Y, cat_encoder, categorical_features
        )
        this_city_X_upsampled, this_city_Y_upsampled = split_X_Y(
            this_city_upsampled_X_Y
        )

        X_upsampled = pd.concat(
            [X_upsampled, this_city_X_upsampled], axis=0, ignore_index=True
        )
        Y_upsampled = pd.concat(
            [Y_upsampled, this_city_Y_upsampled], axis=0, ignore_index=True
        )

    return X_upsampled, Y_upsampled


# if __name__ == "__main__":
#     X_upsampled, Y_upsampled = smote_oversample_run_all()
#     print(X_upsampled.head())
#     print(Y_upsampled.head())


# X_upsampled, Y_upsampled = smote_oversample_run_all()
# print(X_upsampled.head())
# print(Y_upsampled.head())
