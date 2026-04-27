import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_data(path: str) -> None:
    X_train = pd.read_csv(path + "X_train.csv").drop(columns=["date"])
    X_test = pd.read_csv(path + "X_test.csv").drop(columns=["date"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        path + "X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        path + "X_test_scaled.csv", index=False
    )
