import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(data_path: str, model_path: str) -> None:

    X_test = pd.read_csv(data_path + "X_test_scaled.csv")
    y_test = pd.read_csv(data_path + "y_test.csv")

    with open(model_path + "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }

    for k, v in scores.items():
        print(f"  {k.upper()}: {v}")

    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    print("Métriques sauvegardées dans metrics/scores.json")

    results = pd.DataFrame({"y_test": y_test.iloc[:, 0], "y_pred": y_pred})
    results.to_csv("data/predictions.csv", index=False)
    print("Prédictions sauvegardées dans data/predictions.csv")
