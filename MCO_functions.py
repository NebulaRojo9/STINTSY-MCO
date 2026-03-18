
import matplotlib.pyplot as plt
import numpy as np # for importing datasets
import math
import pandas as pd

from neural_network import NeuralNetwork
from data_loader import DataLoader
import torch
from torch import optim
import torch.nn as nn

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

def compareChart(mode, n, cat, figwidth, figheight, urban_columns, rural_columns, title):
    # bar profile
    y = np.arange(n) # label location of categories
    bar_height = 0.4 # the height of bars

    # figure initiation
    fig, ax = plt.subplots(figsize=(figwidth,figheight))

    # bar profile
    rects1 = ax.barh(y - 0.2, urban_columns, height=bar_height, label="Urban", color="#0d8ac0")
    rects2 = ax.barh(y + 0.2, rural_columns, height=bar_height, label="Rural", color="#30e0a8")

    # add labels, title, and legend
    if mode=="T":
        ax.set_xlabel("Average monthly expenses in Php", family="monospace", fontsize=14, fontweight="bold")
        ax.set_ylabel("Expense category", family="monospace", fontsize=14, fontweight="bold")

    if mode=="N":
        ax.set_xlabel("Average monthly income in Php", family="monospace", fontsize=14, fontweight="bold")
        ax.set_ylabel("Income category", family="monospace", fontsize=14, fontweight="bold")
    
    ax.invert_yaxis()
    ax.set_title(title, family="impact", fontsize=20)
    ax.set_yticks(y, cat)
    plt.legend()

    #add labels to bars
    ax.bar_label(rects1, padding=3, fmt="{:.2f}")
    ax.bar_label(rects2, padding=3, fmt="{:.2f}")

    plt.show()

def trainNetwork(network: NeuralNetwork, data_loader: DataLoader, 
        optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, max_epochs=300):
    e = 0
    is_converged = False
    previous_loss = 0
    losses = []

    patience = 10  # epochs to wait for improvement
    min_delta = 1e-4  # minimum improvement to reset patience
    best_loss = float('inf')
    epochs_since_improvement = 0

    # For each epoch
    while e < max_epochs and is_converged is not True:
        
        current_epoch_loss = 0
        
        # Get the batch for this epoch.
        X_batch, y_batch = data_loader.get_batch()
        
        # For each batch
        for X, y in zip(X_batch, y_batch):
            X = torch.Tensor(X).float()
            y = torch.Tensor(y).to(torch.long)
            
            # Empty the gradients of the network.
            optimizer.zero_grad()
            
            # Forward propagation
            scores, probabilities = network.forward(X, verbose=False)
            
            # Compute the loss
            loss = criterion(scores, y)
            
            # Backward propagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            current_epoch_loss += loss.item()
        
        average_loss = current_epoch_loss / len(X_batch)
        losses.append(average_loss)

        # Patience
        current_loss = average_loss

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # If we have gone patience epochs without improvement, we can stop
        if epochs_since_improvement >= patience:
            is_converged = True
        
        # Display the average loss per epoch
        print('Epoch:', e + 1, '\tLoss: {:.6f}'.format(average_loss))
        
        if abs(previous_loss - loss.item()) < 1e-4:
            is_converged = True
        else:
            previous_loss = loss.item()
            e += 1
    
    return losses

def evaluateNetwork(model, X_val, y_val):

    model.eval()

    with torch.no_grad():
        # Get the output of the model for the validation set. 
        # We place ", _" after "outputs" because the model's forward 
        # function returns two values, and we only need the first one for evaluation.  
        outputs, _ = model(X_val) 
        preds = torch.argmax(outputs, dim=1)

    # Since dataset is imbalanced, we use balanced accuracy instead of regular accuracy
    # we need to round them to four decimal places to reduce clutter in the output
    accuracy = balanced_accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, average=None)
    class_precisions = {f"class {i}": p for i, p in enumerate(precision)}
    recall = recall_score(y_val, preds, average=None)
    class_recalls = {f"class {i}": r for i, r in enumerate(recall)}
    f1 = f1_score(y_val, preds, average=None)
    class_f1s = {f"class {i}": f for i, f in enumerate(f1)}

    return {
        "accuracy": accuracy,
        "precision": class_precisions,
        "recall": class_recalls,
        "f1": class_f1s
    }