from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/reward_model_1.zip"
    train(model_path)
    test(model_path)