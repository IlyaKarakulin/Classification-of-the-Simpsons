from Model import Classifier
from Model import get_device

device = get_device()
model = Classifier(device)

path_to_model = './meta_data/best.tar'
path_to_test = './dataset/testset'

model.load_model(path_to_model)

model.test(path_to_test)