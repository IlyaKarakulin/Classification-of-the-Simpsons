import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
from torchvision import transforms


class DatasetLoader(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files
        self._len = len(self.files)
        self.label_encoder = LabelEncoder()

        self.labels = [path.parent.name for path in self.files]
        self.label_encoder.fit(self.labels)

        with open('data/label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)

    def __getitem__(self, index):

        x, size = self.load_sample(self.files[index])

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()

        image = image.resize((self.rescale_size, self.rescale_size))

        return image, image.size