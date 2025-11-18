import torch.nn as nn

def mnist_net():
    return nn.Sequential(
        nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def fmnist_net():
    return nn.Sequential(
        # Stem Block
        nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        
        # Block 1
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        # Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        
        # Classifier Head
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )


def svhn_net():
    return nn.Sequential(
        # Hidden 1
        nn.Conv2d(3, 48, kernel_size=5, padding=2),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2, padding=1),
        nn.Dropout(0.2),

        # Hidden 2
        nn.Conv2d(48, 64, kernel_size=5, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1, padding=1),
        nn.Dropout(0.2),

        # Hidden 3
        nn.Conv2d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2, padding=1),
        nn.Dropout(0.2),

        # Hidden 4
        nn.Conv2d(128, 160, kernel_size=5, padding=2),
        nn.BatchNorm2d(160),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1, padding=1),
        nn.Dropout(0.2),

        # Hidden 5
        nn.Conv2d(160, 192, kernel_size=5, padding=2),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2, padding=1),
        nn.Dropout(0.2),

        # Hidden 6
        nn.Conv2d(192, 192, kernel_size=5, padding=2),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1, padding=1),
        nn.Dropout(0.2),

        # Hidden 7
        nn.Conv2d(192, 192, kernel_size=5, padding=2),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2, padding=1),
        nn.Dropout(0.2),

        # Hidden 8
        nn.Conv2d(192, 192, kernel_size=5, padding=2),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1, padding=1),
        nn.Dropout(0.2),

        # Flatten + MLP
        nn.Flatten(),
        nn.Linear(192 * 5 * 5, 3072),
        nn.ReLU(),
        nn.Linear(3072, 3072),
        nn.ReLU()
    )


def arch_selector(architecture):
    selector = {
        "MNIST": mnist_net,
        "FMNIST": fmnist_net,
        "SVHN": svhn_net
    }
    return selector[architecture]