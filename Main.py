from Model import Classifier
from Model import get_device

path_to_train_and_val = "/home/i.karakulin/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
path_to_testset = "/home/i.karakulin/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset"

# path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
# path_to_testset = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset"

device = get_device()
model = Classifier(device)

model.train(
    path_to_train_and_val=path_to_train_and_val,
    num_epoch=20,
    batch_size=256,
    lr=0.0007,
)

# path_to_model = '/home/i.karakulin/Classification-of-the-Simpsons/meta_data/lost.tar'
# model.load_model(path_to_model)

# model.test(path_to_testset)