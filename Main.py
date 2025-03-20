from Model import Classifier

path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
path_to_testset = '/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset'

model = Classifier()

# SimpsonDataset(path_to_train_and_val)

# model.train(path_to_train_and_val)

# model.save_model()

model.load_model()

model.test(path_to_testset)