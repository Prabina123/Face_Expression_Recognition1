import torchvision.datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import linear_model,conv_model
import torch.optim as optim
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import wandb


##### conv packages ######

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Re-Runs",

    # Define a config dictionary object
    config = {
      "num_classes": 7,
    "num_epochs": 60
    }

)

# checking for the device has gpu option or not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### conv pytorch for Resnet ######
#import torchvision.models as models
#model = models.resnet18()
#model.fc = nn.Linear(512, wandb.config['num_classes'])


## LeNet ########
#model = conv_model.LeNet(3,wandb.config['num_classes'])

## MLP ###
model = linear_model.SimpleNet()


#sending model to gpu
model = model.to(device)

# defining the transforms and dataloaders
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='/afs/crc.nd.edu/user/p/psharma3/NNproject/images/train',transform=transform)

# Split the indices in a stratified way -- just doing so such that the train time is less
indices = np.arange(len(train_dataset))
print(len(train_dataset))
train_indices, val_indices = train_test_split(indices, train_size=0.8, stratify=train_dataset.targets)
print("train indices:",len(train_indices))
# Warp into Subsets and DataLoaders
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)

print("subset:",len(train_sampler))
train_loader = DataLoader(train_dataset, shuffle=False, num_workers=2,sampler = train_sampler,batch_size=50)
#val_loader = DataLoader(train_dataset, shuffle=False, num_workers=2,sampler = val_indices, batch_size=50)
val_loader = DataLoader(train_dataset, shuffle=False, num_workers=2,sampler = val_sampler, batch_size=50)

# intializing the loss functions and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(SimpleNet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)


print("Training started:")
for epoch in range(wandb.config['num_epochs']):
    print(f"Epoch {epoch+1} Started")
    loss_epoch = 0
    correct = 0
    total = 0
    for i,data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # recording the loss
        loss_epoch += outputs.shape[0]*loss.item()

    #training accuracy
    acc = (correct/total)*100
    loss_epoch = loss_epoch/len(train_sampler)

    #validation accuracy

    # vaidation
    model.eval()  # handle drop-out/batch norm layers
    val_loss = 0
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        print("validation started")
        for val_data,val_labels in val_loader:
            val_data,val_labels = val_data.to(device),val_labels.to(device)
            val_out = model(val_data)  # only forward pass - NO gradients!!
            val_loss += criterion(val_out, val_labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(val_out.data, 1)
            val_total += val_labels.size(0)
            val_correct += (predicted == val_labels).sum().item()

    # total loss - divide by number of batches
    val_loss = val_loss / len(val_loader)
    # validation accuracy
    val_acc = (val_correct / val_total) * 100
    print("validation total",val_total)
    #log metrics to wandb
    wandb.log({"Epoch": epoch+1,
               "Training Accuracy": acc,
               "Training Loss": loss_epoch,
               "Validation Accuracy":val_acc,
               "Validation Loss":val_loss})

    print(f"Epoch {epoch + 1} Ended and logged in wandb")


output_path = "../latest_saved_models/latestResNet1.pt"
torch.save(model, output_path)







