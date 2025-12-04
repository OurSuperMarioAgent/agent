from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/reward_model_2.zip"
    train(model_path, 1e5)
    test(model_path)