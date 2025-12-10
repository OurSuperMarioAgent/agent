from train_model import train
from test_model import test

if __name__ == "__main__":
    model_path = "models/CNN_MLP_policy_optim_normal_model_1.zip"
    train(model_path, 3e6)
    test(model_path)