import pickle
import pandas as pd
from xgboost import XGBRegressor


def train_model(data_path: str, model_path: str) -> None:

    X_train = pd.read_csv(data_path + "X_train_scaled.csv")
    y_train = pd.read_csv(data_path + "y_train.csv")

    with open(model_path + "best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    print(f"Paramètres : {best_params}")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )

    model.fit(X_train, y_train)

    with open(model_path + "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modèle entraîné et sauvegardé dans 'xgb_model.pkl'")
