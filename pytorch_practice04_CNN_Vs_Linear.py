# PyTorch Practice 04 - Convolution Vs. Linear, FashionMNIST

# imports
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

# setting seed to have fixed randomness
RANDOM_SEED = torch.manual_seed(42)
RANDOM_SEED_CUDA = torch.cuda.manual_seed(42)

# setting device to "cuda" if able
device = "cuda" if torch.cuda.is_available() else "cpu"

# printing out the time based from timer
def print_time(start: float, end: float, device: torch.device = None):
    # total time is the end time minus the start
    total_time = end - start
    # print info
    print(f"\nTrain time for \"{device}\": {total_time:.3f} seconds")

# allowing for images that are being used to be seen
def visual_image(image, label=None, class_names=None):
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()

# method used to determine accuracy
def accuracy_func(y_true, y_pred):
    # calculating which values are correct
    correct = torch.eq(y_true, y_pred).sum().item()
    # prefroming accuracy formula
    acc = (correct/len(y_pred)) * 100
    # return accuracy
    return acc

# method that returns a dictionary containg the results of a model via dataloader
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_func
            ):
    # initalize variables
    loss, acc = 0, 0
    # test mode
    model.eval() 
    # in inference mode...
    with torch.inference_mode():
        # Accumulate loss and acc per batch
        for X, y in data_loader:
            X, y = X.to(device), y.to(device) # move to device
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_func(y, y_pred.argmax(dim=1))
        # scale loss and acc to find average per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    # return dictionary
    return {"model_name": model.__class__.__name__,
            "model loss": loss.item(),
            "model acc": acc
            }

def train_mode(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer,
               accuracy_func):
    train_loss, train_acc = 0, 0

    model.train() # training mode

    for X, y in data_loader:
        X, y = X.to(device), y.to(device) # move to device

        y_pred = model(X) # forward pass
        loss = loss_fn(y_pred, y) # loss for one batch
        train_loss += loss # accumulate loss
        train_acc += accuracy_func(y, y_pred.argmax(dim=1)) # accumulate acc

        optimizer.zero_grad() # zero gradients
        loss.backward() # backprop
        optimizer.step() # step in optimizer

    # average loss and acc over all batches
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

# model_01 creation, linear using nn.Sequential()
class model_01(nn.Module):
    # initalize the class and model
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # creating layer (Note: adding ReLU() tanks accuracy, needs to be linear for this model) 
        self.model_layers = nn.Sequential(
            nn.Flatten(), # flatten data to turn tensor to vector
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    # forward stepping
    def forward(self, x: torch.Tensor):
        return self.model_layers(x)

# model_02 creation, using CNN
class model_02(nn.Module):
    # initalize the class and model
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convolution block 1
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(), # Re.LU() layer
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(), # Re.LU() layer
            nn.MaxPool2d(kernel_size=2) # max pooling (taking max size of kernel size window)
        )
        # convolution block 2
        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(), # Re.LU() layer
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(), # Re.LU() layer
            nn.MaxPool2d(kernel_size=2) # max pooling (taking max size of kernel size window)
        )
        # classifier layer meant to give the answer a class 
        self.classifier_layer = nn.Sequential(
            nn.Flatten(), # flatten to turn tensor to a vector
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
    # forward stepping
    def forward(self, x: torch.Tensor):
        return self.classifier_layer(self.conv_block_02(self.conv_block_01(x)))

# main method
def main():
# --- Getting Data ---

# getting the training data (will download MNIST if not already)
    train_data = datasets.FashionMNIST(
        # where to download (change to prefered location)
        root="FashionMNIST",
        # getting training data and not testing
        train=True,
        # download the data
        download=True,
        # turn the data into tensors
        transform=ToTensor(),
        # do not tranform the labels
        target_transform=None)
    
# getting the testing data (will download MNIST if not already)
    test_data = datasets.FashionMNIST(
        # where to download (change to prefered location)
        root="FashionMNIST",
        # getting testing data
        train=False,
        # download the data
        download=True,
        # turn the data into tensors
        transform=ToTensor(),
        # do not tranform the labels
        target_transform=None)
    
    # setting up class names for potential use later
    class_names = train_data.classes

# --- Data Loader ---

    # batch size choosen is 64, shuffle is True for train, testing is kept in same order
    BATCH_SIZE = 64

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- Initalizing Models ---

    model01 = model_01(
        input_shape=28*28, # matching the shape to flatten
        hidden_units=10,
        output_shape=len(class_names)
                     ).to(device)
    
    model02 = model_02(
        input_shape=1, # only one color channel
        hidden_units=10,
        output_shape=len(class_names)
                     ).to(device)

    
# --- Loss and Optimizer ---

    # hyperparam for learning rate
    LEARNING_RATE = 0.01

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model01.parameters(), lr=LEARNING_RATE)


# --- MODEL01 Training and Test loop --- 

    # start time for training loop
    start_time = timer()

    # Hyperparam epochs
    EPOCHS = 10

    # loop through epochs
    for epoch in tqdm(range(EPOCHS)):

# --- Training ---

        train_loss, train_acc = train_mode(model=model01,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_func=accuracy_func)

# --- Test ---

        model01_results = eval_model(model=model01, 
                                    data_loader=test_dataloader,
                                    loss_fn=loss_fn, 
                                    accuracy_func=accuracy_func)
        
        # recording answers from test through the dictionary (the dictionary is importnat for comparision)
        test_loss = model01_results["model loss"]
        test_acc = model01_results["model acc"]

        # printing what is happening
        print(f"\nEpoch: {epoch} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    end_time = timer() # end time for training loop
    # print results for speed
    print_time(start_time, end_time, device=str(next(model01.parameters()).device))

    # getting complete average accuracy between train and test
    model01_average_acc = (train_acc + test_acc)/2

    print("\n\nStarting convolution...\n")

# --- MODEL02 Training and Test loop ---

    # adapting optimizer for model02 params
    optimizer = torch.optim.SGD(params=model02.parameters(), lr=LEARNING_RATE)

    # start time for training loop
    start_time = timer()

# --- Training ---

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = train_mode(model=model02,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_func=accuracy_func)

# --- Test ---

        model01_results = eval_model(model=model02, 
                                    data_loader=test_dataloader,
                                    loss_fn=loss_fn, 
                                    accuracy_func=accuracy_func)
        
        # recording answers from test through the dictionary (the dictionary is importnat for comparision)
        test_loss = model01_results["model loss"]
        test_acc = model01_results["model acc"]

        print(f"\nEpoch: {epoch} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    end_time = timer() # end time for training loop
    # print results for speed
    print_time(start_time, end_time, device=str(next(model02.parameters()).device))

    # getting complete average accuracy between train and test
    model02_average_acc = (train_acc + test_acc)/2

    # Print the total accuracy and the percent error
    print(f"\nModel 01 average acc: {model01_average_acc:.3f}, Model 02 average acc: {model02_average_acc:.3f}")
    # CNN, or model02, will preform better than linear assuming hyperparams don't change
    print(f"\nModel 02 preformed {((model02_average_acc - model01_average_acc)/model01_average_acc) * 100:.2f}% better than Model 01")
    # exit code "0"
    exit(0)

# run main
if __name__ == "__main__":
    main()