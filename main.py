from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/mario_ppo_model.zip"
    train(model_path)
    test(model_path)