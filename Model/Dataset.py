import os
import pickle

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms
from pathlib import Path


class SimpsonDataset(Dataset):
    def __init__(self, path_to_files, mode='train'):
        super().__init__()

        self.mode = mode
        self.path_to_files = Path(path_to_files)

        IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

        self.files = [
            p for p in self.path_to_files.rglob('*')
            if p.suffix.lower() in IMAGE_EXTS
        ]

        self._len = len(self.files)


        if mode == 'train':
            self.label_encoder = LabelEncoder()

            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            os.makedirs('./meta_data', exist_ok=True)
            with open('./meta_data/label_encoder.pkl', 'wb') as le_conf:
                pickle.dump(self.label_encoder, le_conf)

        elif mode == 'test' or mode == 'val':
            self.label_encoder = pickle.load(open("./meta_data/label_encoder.pkl", 'rb'))
            self.labels = [path.parent.name for path in self.files]

        else:
            pass

    def __load_sample(self, file):
        image = Image.open(file).convert('RGB') 
        image.load()

        return image, image.size
    
    def __getitem__(self, index):

        x, _ = self.__load_sample(self.files[index])

        transforms_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        x = transforms_img(x)

        label_id = self.label_encoder.transform([self.labels[index]])
        y = label_id.item()
        return x, y
        
    def __len__(self):
        return self._len
    