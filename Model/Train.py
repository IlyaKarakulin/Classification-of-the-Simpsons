from Model import Classifier
from Model import get_device

path_to_train = "./dataset/train"
path_to_val = "./dataset/val"
path_to_test = "./dataset/test"

device = get_device()
model = Classifier(device)

# You can start learning from a pre-trained model
# path_to_model = '../meta_data/best.tar'
# model.load_model(path_to_model)

model.train(
    path_to_train=path_to_train,
    path_to_val=path_to_val,
    num_epoch=60,
    batch_size=128,
    lr=0.003
)