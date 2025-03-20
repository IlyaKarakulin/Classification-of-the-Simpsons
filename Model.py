import os

import torch.nn as nn
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

from Dataset import SimpsonDataset


class Model(nn.Module):
    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=11, stride=4),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, n_classes),
        )

        self.SoftMax = nn.Sequential(
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.SoftMax(x)

        return x


class Classifier():
    def __init__(self):
        self.model = Model()


    def train(self, path_to_train_and_val: str, num_epoch=1, batch_size=32, lr=0.001, save_each_epoch=True):

        test_val_dataset = SimpsonDataset(path_to_train_and_val)

        train_size = int(0.8 * len(test_val_dataset))
        val_size = len(test_val_dataset) - train_size

        train_dataset, val_dataset = random_split(test_val_dataset, [train_size, val_size])
        Simpson_dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        Simpson_dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()

        metrics = []

        for epoch in range(num_epoch):
            print("Epoch:", epoch)

            for x_train, y_train in tqdm(Simpson_dataloader_train, position=0, leave=True):

                optimizer.zero_grad()

                predict = self.model(x_train)
                loss = loss_func(predict, y_train)

                loss.backward()
                optimizer.step()

            metrics.append(self.count_metrics(Simpson_dataloader_val))

            if(save_each_epoch):
                name = f"epoch{epoch}"
                self.__save_model__(name)

        data_metrics = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
        data_metrics.to_csv('./meta_data/metrics.csv')


    def count_metrics(self, dataloader):
        self.model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x_test, y_test in tqdm(dataloader, position=0, leave=True):
                outputs = self.model(x_test)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        avg_accuracy = accuracy_score(all_labels, all_preds)
        avg_precision = precision_score(all_labels, all_preds, zero_division=True, average='macro')
        avg_recall = recall_score(all_labels, all_preds, zero_division=True, average='macro')
        avg_f1 = f1_score(all_labels, all_preds, zero_division=True, average='macro')

        print(f"Avg metrics | Acc: {avg_accuracy:.3f} | P: {avg_precision:.3f} | R: {avg_recall:.3f} | F1: {avg_f1:.3f}")

        self.model.train()
        metric = [avg_accuracy, avg_precision, avg_recall, avg_f1]
        return metric


    def __save_model__(self, name):
        state = self.model.state_dict()
        os.makedirs('./meta_data', exist_ok=True)
        torch.save(state, f"./meta_data/{name}.tar")


    def load_model(self, path="./meta_data/state.tar"):
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

        unique_classes = np.unique(all_labels)

        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
                
        class_metrics = pd.DataFrame({
            'Class': unique_classes,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1': f1_per_class
        })
        
        class_metrics.to_csv('./meta_data/class_metrics_on_testset.csv', index=False)


