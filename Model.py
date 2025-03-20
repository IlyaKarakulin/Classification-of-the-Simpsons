import os

import torch.nn as nn
import torch
from tqdm import tqdm
from Dataset import SimpsonDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Model(nn.Module):
    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=42, kernel_size=256, stride=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return x


class Classifier():
    def __init__(self):
        self.model = Model()

    def train(self, path_to_train_and_val: str, num_epoch=1, batch_size=32, lr=0.001):

        test_val_dataset = SimpsonDataset(path_to_train_and_val)

        train_size = int(0.8 * len(test_val_dataset))
        val_size = len(test_val_dataset) - train_size

        train_dataset, val_dataset = random_split(test_val_dataset, [train_size, val_size])
        Simpson_dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()

        for _ in range(num_epoch):

            for x_train, y_train in tqdm(Simpson_dataloader_train, position=0, leave=True):

                optimizer.zero_grad()

                predict = self.model(x_train)
                loss = loss_func(predict, y_train)

                loss.backward()
                optimizer.step()


    def save_model(self):
        state = self.model.state_dict()
        os.makedirs('./Classification-of-the-Simpsons/meta_data', exist_ok=True)
        torch.save(state, "./Classification-of-the-Simpsons/meta_data/state.tar")


    def load_model(self, path="./Classification-of-the-Simpsons/meta_data/state.tar"):
        self.model = Model()
        
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()


    def test(self, path_to_test_data: str, batch_size=32):
        self.model.eval()
        
        testset = SimpsonDataset(path_to_test_data, mode='test')
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader, position=0, leave=True):
                outputs = self.model(x_test)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


