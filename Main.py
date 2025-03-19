import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import tqdm


from Dataset import SimpsonDataset
from Model import model


path_to_train_and_val = "/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset/simpsons_dataset"
dataset = SimpsonDataset(path_to_train_and_val)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
Simpson_dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
loss_func = nn.CrossEntropyLoss()

model.train()

num_epoch = 5

for _ in range(num_epoch):

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for x_train, y_train in tqdm(Simpson_dataloader_train, position=0, leave=True):
        loss_func.zero_grad()

        predict = model(x_train)
        loss = loss_func(predict, y_train)
        loss.backward()
        loss_func.step()

state = model.state_dict()
torch.save(state, "state.tar")


path_to_test = '/home/ilya/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/kaggle_simpson_testset/kaggle_simpson_testset'
test_loader = DataLoader(path_to_test)

# with torch.no_grad():
#     logits = []

#     for inputs in test_loader:
#         inputs = inputs.to(self.device)
#         self.model.eval()
#         outputs = self.model(inputs).cpu()
#         logits.append(outputs)

# probabilities = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
