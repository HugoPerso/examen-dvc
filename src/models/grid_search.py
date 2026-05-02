import pickle
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def grid_search(data_path: str, model_path: str, param_grid: dict) -> None:

    X_train = pd.read_csv(data_path + "X_train_scaled.csv")
    y_train = pd.read_csv(data_path + "y_train.csv")

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    with open(model_path + "best_params.pkl", "wb") as f:
        pickle.dump(grid_search.best_params_, f)
    print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")


if __name__ == "__main__":
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    }
    grid_search("data/processed_data/", "models/", param_grid)
