from Model import Classifier
from Model import get_device

path_to_train_and_val = "/home/i.karakulin/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
path_to_testset = "/home/i.karakulin/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset"

device = get_device()
model = Classifier(device)

model.train(
    path_to_train_and_val=path_to_train_and_val,
    num_epoch=1,
    batch_size=256,
    lr=0.001,
    save_each_epoch=True
)

model.load_model('/home/i.karakulin/Classification-of-the-Simpsons/meta_data/epoch0.tar')

model.test(path_to_testset)