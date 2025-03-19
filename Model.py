import torch.nn as nn
import torch
from tqdm import tqdm
from Dataset import SimpsonDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim


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



# path_to_test = '/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/kaggle_simpson_testset'
# test_loader = DataLoader(path_to_test)

# with torch.no_grad():
#     logits = []

#     for inputs in test_loader:
#         inputs = inputs.to(self.device)
#         self.model.eval()
#         outputs = self.model(inputs).cpu()
#         logits.append(outputs)

# probabilities = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()


class Classifier():
    def __init__(self):
        self.model = Model()

    def train(self, path_to_train_and_val, num_epoch=1, batch_size=32, lr=0.001):

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

        state = self.model.state_dict()
        torch.save(state, "state.tar")

        self.model.eval()

    def test(path_to_test_data):
        
        testset = SimpsonDataset(path_to_test_data, mode='test')
        Simpson_dataloader_train = DataLoader(testset, batch_size=batch_size, shuffle=True)

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

        state = self.model.state_dict()
        torch.save(state, "state.tar")

        self.model.eval()

