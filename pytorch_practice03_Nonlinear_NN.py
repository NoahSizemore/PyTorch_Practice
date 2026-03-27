# PyTorch Project 03: NN Classification

import requests
from pathlib import Path
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# impacting torch code only, need to set state/seed for sklearn code
RANDOM_SEED = torch.manual_seed(42)
RANDOM_SEED_CUDA = torch.cuda.manual_seed(42)

# allowing for gpu computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# ***--- Creating the model (replaced by model_01) ---***
"""
class CircleMode01(nn.Module):
    # initalizer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # creating two nn.Linear() layers to handel shape of data, 2:1
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # input layer 2:5
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # output layer 5:1

    # define forward methoid for forward pass
    def forward(self, x):
        # x, into layer_1, into layer_2, into the output.
        return(self.layer_2(self.layer1(x))) 
"""  
# graphing the initial data from the df
def pd_base(X0, X1, y):
    # creating figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x=X0, y=X1, c=y, cmap=plt.cm.RdYlBu)
    plt.title("Initial Data")

# method used to determine accuracy
def accuracy_func(y_true, y_pred):
    # calculating which values are correct
    correct = torch.eq(y_true, y_pred).sum().item()
    # prefroming accuracy formula
    acc = (correct/len(y_pred)) * 100
    # return accuracy
    return acc

# ***Source for plot_predictions and plot_decision_boundary (https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py)
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

# method for plotting model_01 training
def plot_boundary_train(model_01, X_train, y_train):
    # creating figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.title("Train")
    # implementing plot_decision_boundary for most of graphing
    plot_decision_boundary(model_01, X_train, y_train)

# method for plotting model_01 testing 
def plot_boundary_test(model_01, X_test, y_test):
    # creating figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 3)
    plt.title("Test")
    # implementing plot_decision_boundary for most of graphing
    plot_decision_boundary(model_01, X_test, y_test)

# main method
def main():
    # sklearn dataset
    n_samples = 1000 # number of samples
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    # seperating the logic data to be used easier
    X0 = X[:, 0]
    X1 = X[:, 1]

    # creating the df 
    circles = pd.DataFrame({"X1: ": X0,"X2: ": X1, "label": y})

# --- Creating tensors ---
    # both are numpy arrays meaning that the need to use "from_numpy" 
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

# ***--- Splitting into training and test ---***
    # splitting randomly due to the data, test size makes it more-or-less an 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Initalizing the model(s) ---
    # model_00 = CircleMode01().to(device) # replaced by model_01

    # model_00 is the exact code implemented by nn.Sequential(), model_01 makes it easier
    model_01 = nn.Sequential(
        # adding nn.ReLU() allows for the fit to be non-linear
        nn.Linear(in_features=2, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=1)
    ).to(device)

# ***--- Loss and optimizer ---***
    # loss measures how wrong a model is
    loss_fn = nn.BCEWithLogitsLoss() # includes Sigmoid acitvation function built-in
    optimizer = torch.optim.SGD(params=model_01.parameters(), lr=0.01) # SGD to optimize

# ***--- Training and testing loop ---***
    epochs = 3000 # number of loops through data

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    # looping for epochs
    for epoch in range(epochs):

# --- Training loop ---
        # put model in training mode
        model_01.train()

         # do the forward pass
        y_logits = model_01(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # turning logits to pred probs into pred labels
        # Calculate the loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCE w/ Logits loss expects raw logits as input
        loss_acc = accuracy_func(y_true=y_train, y_pred=y_pred) # using accuracy function
        # optimizer zero grad
        optimizer.zero_grad()
        # loss backward (backpropagation)
        loss.backward()
        # optimizer step (grad descent)
        optimizer.step()

# --- Testing loop ---
        # put model in testing mode
        model_01.eval()
        # infernce mode for stability
        with torch.inference_mode():
            # test forward pass
            test_logits = model_01(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # calculate the loss and accuracy
            test_loss = loss_fn(test_logits, y_test) # BCE w/ Logits loss expects raw logits as input
            test_accuracy = accuracy_func(y_true=y_test, y_pred=test_pred) # using accuracy function
        # printing what is happening every 10 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.5f}, Acc: {loss_acc:.2f}%, Test loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}%")


# ***--- Visualizations ---***

    pd_base(X0=X0, X1=X1, y=y) # initial data
    plot_boundary_train(model_01, X_train, y_train) # plotting boundary for model_01 training data
    plot_boundary_train(model_01, X_test, y_test) # plotting boundary for model_01 testing data
    # show plots
    plt.show()

# run main
if __name__ == "__main__":
    main()

