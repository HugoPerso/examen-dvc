import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(source_path: str, dest_path: str) -> None:

    df_data = pd.read_csv(source_path)
    X = df_data.drop(columns=["silica_concentrate"])
    y = df_data["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pd.DataFrame(X_train).to_csv(dest_path + "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(dest_path + "X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(dest_path + "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(dest_path + "y_test.csv", index=False)
