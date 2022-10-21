def rolling_dataframe_baby(df, n=2):
    df_rolled = df.rolling(n).mean()
    df_rolled.loc[:, "year"] = df.loc[:, "year"]
    df_rolled.loc[:, "weekofyear"] = df.loc[:, "weekofyear"]
    return df_rolled


def lagging_columns(df):
    df_rolled = df.rolling(n).mean()
    df_rolled.loc[:, "year"] = df.loc[:, "year"]
    df_rolled.loc[:, "weekofyear"] = df.loc[:, "weekofyear"]
    return df_rolled