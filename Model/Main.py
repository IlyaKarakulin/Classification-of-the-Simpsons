from Model import Classifier
from Model import get_device

path_to_train_and_val = "/home/i.karakulin/Classification-of-the-Simpsons/simpsons dataset/data/dataset"
path_to_testset = "/home/i.karakulin/Classification-of-the-Simpsons/simpsons dataset/data/testset"

device = get_device()
model = Classifier(device)


# model.train(
#     path_to_train_and_val=path_to_train_and_val,
#     num_epoch=15,
#     batch_size=512,
#     lr=0.0005,
# )

# path_to_model = '/home/i.karakulin/Classification-of-the-Simpsons/meta_data/10.tar'

# model.load_model(path_to_model)

# model.test(path_to_testset)