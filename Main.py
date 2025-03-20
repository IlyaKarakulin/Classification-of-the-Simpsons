from Model import Classifier

path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
path_to_testset = '/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/prepare_testset'

model = Classifier()

model.train(
    path_to_train_and_val=path_to_train_and_val,
    num_epoch=3,
    batch_size=64,
    lr=0.001,
    save_each_epoch=True
)

# model.load_model()



# model.test(path_to_testset)