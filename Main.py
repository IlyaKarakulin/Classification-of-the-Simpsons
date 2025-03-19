from Model import Classifier

path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"

model = Classifier()
model.train(path_to_train_and_val)

