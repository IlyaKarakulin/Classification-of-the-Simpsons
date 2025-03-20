import os

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
from torchvision import transforms
from pathlib import Path


class SimpsonDataset(Dataset):
    def __init__(self, path_to_files, mode='train'):
        super().__init__()

        self.mode = mode
        self.path_to_files = Path(path_to_files)
        self.files = list(self.path_to_files.rglob('*.jpg'))
        self._len = len(self.files)


        if mode == 'train' or mode == 'val':
            self.label_encoder = LabelEncoder()

            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            os.makedirs('./Classification-of-the-Simpsons/meta_data', exist_ok=True)
            with open('./Classification-of-the-Simpsons/meta_data/label_encoder.pkl', 'wb') as le_conf:
                pickle.dump(self.label_encoder, le_conf)

        elif mode == 'test':
            self.label_encoder = pickle.load(open("./Classification-of-the-Simpsons/meta_data/label_encoder.pkl", 'rb'))
            self.labels = [path.parent.name for path in self.files]
            
        else:
            pass

    def load_sample(self, file):
        image = Image.open(file)
        image.load()

        return image, image.size
    
    def __getitem__(self, index):

        x, _ = self.load_sample(self.files[index])

        transforms_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        x = transforms_img(x)

        label_id = self.label_encoder.transform([self.labels[index]])
        y = label_id.item()
        return x, y
        
    def __len__(self):
        return self._len
    