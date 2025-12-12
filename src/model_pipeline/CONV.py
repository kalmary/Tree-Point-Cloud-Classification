import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn.functional import dropout
from torchvision.models import resnet18, resnet34

class TreeClassifierCNN(nn.Module):

    def __init__(self, num_channels, num_classes):
        super(TreeClassifierCNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=3)

        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)


        self.flatten = nn.Flatten()


        # Fully Connected Layers
        self.fc1 = nn.Linear(61952, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.conv1(self.relu(x))
        x = self.conv2(self.relu(x))
        x = self.pool1(x)

        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class ResNetTreeClassifier(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate=0.5):
        super(ResNetTreeClassifier, self).__init__()
        # Load ResNet34 without pretrained weights
        self.base_model = resnet34()

        # Modify the first conv layer to accept 'num_channels' input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=3,
            bias=False
        )

        # Store the original fully connected layer to re-purpose its input features
        num_ftrs = self.base_model.fc.in_features

        # Remove the original final fully connected layer
        self.base_model.fc = nn.Identity()  # Replace with an identity layer to get features before final FC

        # Add a dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Define a new final fully connected layer
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Pass input through the base ResNet model up to the point before the original FC layer
        features = self.base_model(x)

        # Apply dropout to the features
        features = self.dropout(features)

        # Pass the dropout-affected features through the new classifier layer
        out = self.classifier(features)
        return out




def main():
    num_channels = 5
    num_classes = 33

    input_dim = (350, 350)

    dummy_input = torch.randn(4, num_channels, input_dim[0], input_dim[1])

    model = ResNetTreeClassifier(num_channels, num_classes)
    model = TreeClassifierCNN(num_channels, num_classes)
    model.eval()
    out = model(dummy_input)

    print("Model output shape:", out.shape)

if __name__ == '__main__':
    main()
