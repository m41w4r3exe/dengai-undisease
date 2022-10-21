from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
from sklearn.base import TransformerMixin, BaseEstimator


warnings.filterwarnings("ignore")


class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Put a breakpoint to below return line to debug transformed values
        # TODO:make plots of transformed and non transformed data
        return data

    def fit(self, data, y=None, **fit_params):

        return self


def best_pipeline_intown(trainX):  ### returns predicted Y

    numerical_features = list(
        trainX.select_dtypes(include=["int64", "float64"]).columns
    )
    categorical_features = list(trainX.select_dtypes(include=[object]).columns)

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            (
                "debugger",
                Debugger(),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder()

    # Bundle Preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=100)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            # (
            #     "debugger",
            #     Debugger(),
            # ),
            ("model", model),
        ]
    )

    return pipeline
