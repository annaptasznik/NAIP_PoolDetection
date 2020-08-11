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



def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

def get_example_per_class(loader, class_names):
    # get example of each class of training data
    f=0
    while(f<100):
        # Get a batch of training data
        inputs, classes = next(iter(loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        if set(class_names[x] for x in classes) == set(class_names):
            f=100
        else:
            f+=1
    imshow(out, title=[class_names[x] for x in classes])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution Layer
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Convolution Layer
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        
        # FC layer
        self.fc1 = nn.Linear(32 * 6 * 6, 5)
    
        
    def forward(self, x):
        # define the data flow through the deep learning layers
        x = self.pool(F.relu(self.l1(x))) # 16x16 x 14 x 14
        x = self.pool(F.relu(self.l2(x))) # 16x32x6x6
        # print(x.shape)
        x = x.reshape(-1, 32*6*6) # [16 x 1152]# CRUCIAL: 
        # print(x.shape)
        x = self.fc1(x)
        return x

'''
Optimize, calculate gradient, etc for each epoch
'''
def mod(optimizer, pred, x, y):
        optimizer.zero_grad() # clear (reset) the gradient for the optimizer
        pred = m(x)
        loss = criterion(pred, y)
        loss.backward() # calculating the gradient
        optimizer.step() # backpropagation: optimize the model

'''
Get confusion matrix values
'''
def get_accuracy_breakdown(all_gt, all_pred):
    result_count = [[0, 0, 0, 0] ,[0, 0, 0, 0] ,[0, 0, 0, 0] ,[0, 0, 0, 0] ]
    index = 0
    while(index < len(all_gt)):
        gt = all_gt[index]
        prediction = all_pred[index]
        result_count[gt][prediction] += 1
        index += 1
    return result_count



shuff = True
start = datetime.datetime.now()
# transform to do random affine and cast image to PyTorch tensor
trans_ = torchvision.transforms.Compose(
    [
    # torchvision.transforms.RandomAffine(10),
    torchvision.transforms.ToTensor()]
)

bs = 3


'''
Load test data
'''
# Setup the dataset
ds = torchvision.datasets.ImageFolder(train_path, transform=trans_)

# Setup the dataloader
loader = torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=shuff)

for x, y in loader:
    break


class_names = ["lawn","pool","roof","street"]

'''
Create and train the model
'''
m = CNN()
pred = m(x)
#print(pred.shape)


# training
criterion = nn.CrossEntropyLoss()
num_epoches = 10

for epoch_id in range(num_epoches):
    optimizer = optim.SGD(m.parameters(), lr=0.01 * 0.95 ** epoch_id)
    for x, y in tqdm.tqdm(loader):
        mod(optimizer, pred, x, y)

'''
Load test data and run samples through the model
'''
# Setup the dataset #
test_ds = torchvision.datasets.ImageFolder(test_path,transform=trans_)

# Setup the dataloader
testloader = torch.utils.data.DataLoader(test_ds,batch_size=bs, shuffle=shuff)

all_gt = []
all_pred = []


for x, y in tqdm.tqdm(testloader):
    optimizer.zero_grad() # clear (reset) the gradient for the optimizer
    all_gt += list(y.numpy().reshape(-1))
    pred = torch.argmax(m(x), dim=1)
    all_pred += list(pred.numpy().reshape(-1))

    actual = (y.numpy().reshape(-1))
    prediction = (pred.numpy().reshape(-1))

'''
Get ground truth and predictions for entire dataset
'''
all_gtT = []
all_predT = []
for x, y in tqdm.tqdm(loader):
    optimizer.zero_grad() # clear (reset) the gradient for the optimizer
    all_gtT += list(y.numpy().reshape(-1))
    pred = torch.argmax(m(x), dim=1)
    all_predT += list(pred.numpy().reshape(-1))


'''
Display results
'''
print("Accuracy Chart: ")
print(get_accuracy_breakdown(all_gt, all_pred))

#print(all_gt)
#print(all_pred)

print(len(all_gt))
print(len(all_pred))

acc = np.sum(np.array(all_gt) == np.array(all_pred)) / len(all_gt)
print("Testing accuracy is:", acc)

accT = np.sum(np.array(all_gtT) == np.array(all_predT)) / len(all_gtT)
print("Training accuracy is:", accT)
end = datetime.datetime.now()

print(end-start)
