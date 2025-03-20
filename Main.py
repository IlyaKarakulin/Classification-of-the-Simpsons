from Model import Classifier
import torch

path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
path_to_testset = '/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset'

if torch.cuda.is_available():
    device = torch.device("cuda:3")
    print(f"Using GPU: {torch.cuda.get_device_name(3)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


model = Classifier(device)

model.train(
    path_to_train_and_val=path_to_train_and_val,
    num_epoch=3,
    batch_size=64,
    lr=0.001,
    save_each_epoch=True
)

# model.load_model()

# model.test(path_to_testset)