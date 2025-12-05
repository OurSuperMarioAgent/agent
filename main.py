from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/reward_model_2_11.zip"
    train(model_path, 3e4, "models/reward_model_2_10.zip")
    test(model_path)