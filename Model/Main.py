from Model import Classifier
from Model import get_device

path_to_train = "./dataset/train"
path_to_val = "./dataset/val"
path_to_test = "./dataset/test"

device = get_device()
model = Classifier(device)


model.train(
    path_to_train=path_to_train,
    path_to_val=path_to_val,
    num_epoch=15,
    batch_size=512,
    lr=0.0005
)

# path_to_model = '/home/i.karakulin/Classification-of-the-Simpsons/meta_data/10.tar'

# model.load_model(path_to_model)

# model.test(path_to_testset)