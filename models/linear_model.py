import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1500)
        self.fc2 = nn.Linear(1500, 750)
        self.fc3 = nn.Linear(750, 375)
        self.fc4 = nn.Linear(375, 180)
        self.fc5 = nn.Linear(180, 90)
        self.fc6 = nn.Linear(90, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # reshape the input tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        return x
