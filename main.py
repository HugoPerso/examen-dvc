from src.data.split_data import split_data
from src.data.normalize_data import normalize_data
from src.models.grid_search import grid_search
from src.models.train import train_model
from src.models.evaluate import evaluate_model

raw_path = "data/raw_data/raw.csv"
preprocess_path = "data/processed_data/"
model_path = "models/"

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

if __name__ == "__main__":
    split_data(source_path=raw_path, dest_path=preprocess_path)
    normalize_data(path=preprocess_path)
    grid_search(data_path=preprocess_path, model_path=model_path, param_grid=param_grid)
    train_model(data_path=preprocess_path, model_path=model_path)
    evaluate_model(data_path=preprocess_path, model_path=model_path)
