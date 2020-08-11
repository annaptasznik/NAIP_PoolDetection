import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pdb
import datetime


train_path = r'C:/Users\Anna Ptasznik\Desktop/poolfinder\data/resized_images/train'
test_path = r'C:/Users\Anna Ptasznik\Desktop/poolfinder\data/resized_images/test'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        self.fc1 = nn.Linear(32 * 6 * 6, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.l1(x))) 
        x = self.pool(F.relu(self.l2(x))) 
        x = x.reshape(-1, 32*6*6)
        x = self.fc1(x)
        return x

'''
Optimize, calculate gradient, etc for each epoch
'''
def mod(optimizer, pred, x, y):
        optimizer.zero_grad()
        pred = m(x)
        loss = criterion(pred, y)
        loss.backward() 
        optimizer.step() 

epochs = [2,3,5,10,15,20,30]
batches = [2,3,5,20,50,150]

max_testing_acc = 0

ep_list = []

for ep in epochs:
    shuff = True
    start = datetime.datetime.now()
    trans_ = torchvision.transforms.Compose(
        [
        torchvision.transforms.ToTensor()]
    )

    bs = 5
    shuff = True

    # Setup the dataset
    ds = torchvision.datasets.ImageFolder(train_path, transform=trans_)

    for batch in batches:
        bs = batch
        # Setup the dataloader
        loader = torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=shuff)

        for x, y in loader:
            break

        m = CNN()
        pred = m(x)
        
        # training
        criterion = nn.CrossEntropyLoss()
        num_epoches = ep

        for epoch_id in range(num_epoches):
            optimizer = optim.SGD(m.parameters(), lr=0.01 * 0.95 ** epoch_id)
            for x, y in tqdm.tqdm(loader):
                mod(optimizer, pred, x, y)
        
        # Setup the dataset #
        test_ds = torchvision.datasets.ImageFolder(test_path,transform=trans_)

        # Setup the dataloader
        testloader = torch.utils.data.DataLoader(test_ds,batch_size=bs, shuffle=shuff)

        all_gt = []
        all_pred = []

        for x, y in tqdm.tqdm(testloader):
            optimizer.zero_grad() 
            all_gt += list(y.numpy().reshape(-1))
            pred = torch.argmax(m(x), dim=1)
            all_pred += list(pred.numpy().reshape(-1))

            actual = (y.numpy().reshape(-1))
            prediction = (pred.numpy().reshape(-1))

        all_gtT = []
        all_predT = []
        for x, y in tqdm.tqdm(loader):
            optimizer.zero_grad()
            all_gtT += list(y.numpy().reshape(-1))
            pred = torch.argmax(m(x), dim=1)
            all_predT += list(pred.numpy().reshape(-1))

        acc = np.sum(np.array(all_gt) == np.array(all_pred)) / len(all_gt)
        accT = np.sum(np.array(all_gtT) == np.array(all_predT)) / len(all_gtT)

        if acc > max_testing_acc:
            max_testing_acc = acc
            torch.save(m(x), 'results/best_model')
            best_batch = bs
            best_epoch = ep
            print(max_testing_acc)

print("___")
print(best_batch)
print(best_epoch)