from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/CNN_optim_model_1.zip"
    train(model_path, 5e5)
    test(model_path)