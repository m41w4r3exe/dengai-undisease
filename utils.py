import pandas as pd


def export_csv_results(pipeline):
    official_testX = pd.read_csv("data/dengue_features_test.csv")
    predsY = pipeline.predict(official_testX)
    preds_df = official_testX.iloc[:, :3]
    preds_df["total_cases"] = predsY.round(0).astype(int)

    # TODO: make automatic version numbering
    preds_df.to_csv("./results/dengue_preds_v1.csv", index=False)
