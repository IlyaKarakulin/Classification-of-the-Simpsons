from Model import Classifier
from Model import get_device

path_to_train = "./dataset/train"
path_to_val = "./dataset/val"
path_to_test = "./dataset/test"

device = get_device()
model = Classifier(device)

# model.train(
#     path_to_train=path_to_train,
#     path_to_val=path_to_val,
#     num_epoch=15,
#     batch_size=64,
#     lr=0.0005
# )

path_to_model = '/home/i.karakulin/Classification-of-the-Simpsons/meta_data/9.tar'
path_to_test = '/home/i.karakulin/Classification-of-the-Simpsons/dataset/testset'

model.load_model(path_to_model)

model.test(path_to_test)