# This project is going over PyTorch workflow

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Global random seed ---
RANDOM_SEED = torch.manual_seed(42)

# ***--- LR model: starts with random values, looks at training data, adjust 
# values to better represent the ideal values ---***
class LinearRegressionModel(nn.Module):

    # initializer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initalizing model parameters
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # method for moving forward in the model, expecting a torch.Tensor in and out
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias # is lr formula
    
# method for plotting predicitions 
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):

# --- Initialize and setup graph ---
    # creating figure
    plt.figure(figsize=(10, 7))

    # plotting training and testing data
    plt.scatter(train_data, train_labels, c="r", s=4, label="Training data plotting")
    plt.scatter(test_data, test_labels, c="b", s=4, label="Test data plotting")

    # only will plot if predictions is not None
    plt.scatter(test_data, predictions, c="g", s=4, label="Predictions") if predictions is not None else None

    # show legend
    plt.legend(prop={"size": 14})

    # display graph
    plt.show()

# method for plotting loss curve
def plot_loss(epoch_count, loss_values, test_loss_values):
    
# --- Initialize and setup graph ---
    # creating figure
    plt.figure(figsize=(10, 7))

    # Train and testing loss
    plt.plot(epoch_count, loss_values, label="Train Loss")
    plt.plot(epoch_count, test_loss_values, label="Test Loss")

    # making the graph visually better
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
        
    # display graph
    plt.show()


# main method
def main():
    # allowing for GPU computation
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ***--- Simple LR program ---***

    # known parameters for grading model
    weight = 0.7 # LR b
    bias = 0.3 # LR a

    start = 0
    end = 1
    step = 0.02
    # capital because X is tensor
    X = torch.arange(start, end, step).unsqueeze(dim=1)

    # LR: y = a + Xb; lowercase because is vector
    y = weight * X + bias

# --- Splitting into training ---
    train_split = int(0.8 * len(X)) # choosing the 80/20 split
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

# --- Initalizing model, loss, and optimizer ---
    model_00 = LinearRegressionModel() # naming model
    loss_fn = nn.L1Loss() # creating loss
    # lr (learning rate) most important hyper-parameter that developer sets, params is parms we want to optimize
    optimizer = torch.optim.SGD(params=model_00.parameters(), lr=0.01) # creaing the optimizer
    
# ***--- Building training and testing loop ---***
    epochs = 200 # one loop through data, hyperparam

    # tracking values 
    epoch_count = []
    loss_values = []
    test_loss_values = []

# --- Training loop ---
    for epoch in range(epochs): # loop through the data

        # for each loop through the data...
        model_00.train() # putting model in "training mode", allows for gradients

        y_pred = model_00(X_train) # forward pass
        loss = loss_fn(y_pred, y_train) # calculate the loss
        optimizer.zero_grad() # optimze zero grad
        loss.backward() # prefrom backpropagation on the loss with respect to the parameters of the model
        optimizer.step() # step the optimizer (grad descent)

        model_00.eval() # take model out of "training mode", testing

        with torch.inference_mode(): # using inference mode to turn off grad tracking
            y_pred = model_00(X_test)
            test_loss = loss_fn(y_pred, y_test) # this is the model before loss and training
    
        if epoch % 10 == 0: 
            print(f"Epoch: {epoch}, Loss: {loss}, Test loss: {test_loss}")
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)

            
# --- Visualization ---

  # using plot method to visualize 
    """
    plot_predictions(X_train, y_train, X_test, y_test) # the non-trained graph
    """

    with torch.inference_mode():
        plot_predictions(X_train, y_train, X_test, y_test, y_pred)
        plot_loss(epoch_count, loss_values, test_loss_values)
     
    exit(0) # exit code "0"

# run main
if __name__ == "__main__":
    main()
