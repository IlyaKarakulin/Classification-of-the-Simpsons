import os

import torch.nn as nn
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

from Dataset import SimpsonDataset

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

class Model(nn.Module):
    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, n_classes),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.size())
        x = self.conv6(x)
        # print(x.size())

        x = x.view(x.size(0), 256)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Classifier():
    def __init__(self, device='cpu'):
        self.device = device
        self.model = Model().to(self.device)

        self.writer = None 

        if self.device == 'cpu':
            self.num_workers = 0
            self.pin_memory = False
        else:
            self.num_workers = 16
            self.pin_memory = True


    def train(self, path_to_train: str, path_to_val: str, num_epoch=100, batch_size=64, lr=0.01):
        train_dataset = SimpsonDataset(path_to_train, mode='train')
        val_dataset = SimpsonDataset(path_to_val, mode='val')

        self.writer = SummaryWriter(f'meta_data/MyModel')

        Simpson_dataloader_train = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
        Simpson_dataloader_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

        loss_func = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.02)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)

        metrics_data = []
        self.model.train()

        best_val_f1 = 0.0

        print(f"Num epoch: {num_epoch} | Batch size: {batch_size} | Initial learning rate: {lr}")

        for count_epoch in range(num_epoch):
            print("Epoch:", count_epoch)

            train_metrics = self.epoch(count_epoch, Simpson_dataloader_train, self.optimizer, loss_func)
            val_metrics = self.validation(Simpson_dataloader_val, loss_func)

            self.scheduler.step(train_metrics['Train_Loss'])
            new_lr = self.optimizer.param_groups[0]['lr']

            self.__log_metrics(train_metrics, val_metrics, count_epoch)
            self.__log_weights(count_epoch)
            
            all_metrics = train_metrics | val_metrics
            metrics_data.append(all_metrics.values())
            
            if(val_metrics["Val_F1"] > best_val_f1):
                self.__save_model("best")
                best_val_f1 = val_metrics["Val_F1"]

            # self.__save_model(f"{count_epoch}")

            print(
                f"Lr: {round(new_lr, 8)} | Loss Train: {train_metrics['Train_Loss']:.3f} | Loss Val: {val_metrics['Val_F1']:.3f} | "
                f"Acc: {val_metrics['Val_Acc']:.3f} | P: {val_metrics['Val_P']:.3f} | "
                f"R: {val_metrics['Val_R']:.3f} | F1: {val_metrics['Val_F1']:.3f}"
            )

        self.writer.close()
        self.__save_model("lost")
        data_metrics = pd.DataFrame(metrics_data, columns=['Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Val_P', 'Val_R', 'Val_F1'])
        data_metrics.to_csv('./meta_data/metrics.csv')


    def epoch(self, count_epoch, train_dataloader, optimizer, loss_func):
        summ_loss = 0.0
        running_corrects = 0.0
        processed_data = 0.0

        for batch_idx, (x_train, y_train) in enumerate(tqdm(train_dataloader, position=0, ncols=75, leave=True)):
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            optimizer.zero_grad()

            predict = self.model(x_train)
            loss = loss_func(predict, y_train)

            loss.backward()

            if batch_idx == len(train_dataloader) - 1:
                self.__log_gradients(count_epoch)

            optimizer.step()

            res_pred = torch.argmax(predict, 1)
            summ_loss += loss.item() * x_train.size(0)
            running_corrects += torch.sum(res_pred == y_train.data)
            processed_data += x_train.size(0)

        train_loss = summ_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data

        metrics = dict(
            {
                "Train_Loss":  train_loss,
                "Train_Acc": train_acc
            }
        )

        return metrics
    
    def __log_metrics(self, train_metrics, val_metrics, epoch):
        for metric, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric}', value, epoch)
        
        for metric, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric}', value, epoch)

    def __log_weights(self, epoch):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def __log_gradients(self, epoch):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f"Gradients/{name}", 
                    param.grad.clone().cpu().data.numpy(), 
                    epoch
                )


    def validation(self, val_dataloader, loss_func):
        self.model.eval()

        all_labels = []
        all_preds = []

        running_loss = 0.0
        count_data = 0.0
        
        for x_val, y_val in val_dataloader:
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

            with torch.no_grad():
                predict = self.model(x_val)
                loss = loss_func(predict, y_val)
                preds = torch.argmax(predict, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())

            running_loss += loss.item() * x_val.size(0)
            count_data += x_val.size(0)

        
        val_loss = running_loss / count_data
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=True, average='macro')
        recall = recall_score(all_labels, all_preds, zero_division=True, average='macro')
        f1 = f1_score(all_labels, all_preds, zero_division=True, average='macro')

        self.model.train()
        metrics = dict(
            {
                "Val_Loss": val_loss,
                "Val_Acc": accuracy,
                "Val_P": precision,
                "Val_R": recall,
                "Val_F1": f1
            }
        )
        return metrics


    def __save_model(self, name):
        state = self.model.state_dict()
        os.makedirs('./meta_data', exist_ok=True)
        torch.save(state, f"./meta_data/{name}.tar")


    def load_model(self, path):
        self.model = Model().to(self.device)
        
        state_dict = torch.load(path, weights_only=True, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        
        self.model.eval()


    def test(self, path_to_test_data: str, batch_size=32):
        print("Testing...")
        self.model.eval()
        
        testset = SimpsonDataset(path_to_test_data, mode='test')
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                outputs = self.model(x_test)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        unique_classes = np.unique(all_labels)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=True, labels=unique_classes)
        
        label_encoder = pickle.load(open("./meta_data/label_encoder.pkl", 'rb'))

        class_metrics = pd.DataFrame({
            'Class': label_encoder.inverse_transform(unique_classes),
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1': f1_per_class
        })
        
        class_metrics.to_csv('./meta_data/test_metric.csv', index=False)

