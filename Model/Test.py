from Model import Classifier
from Model import get_device

path_to_train = "./dataset/train"
path_to_val = "./dataset/val"
path_to_test = "./dataset/test"

device = get_device()
model = Classifier(device)

path_to_model = '/home/i.karakulin/Classification-of-the-Simpsons/meta_data/best.tar'
path_to_test = '/home/i.karakulin/Classification-of-the-Simpsons/dataset/testset'

model.load_model(path_to_model)

model.test(path_to_test)