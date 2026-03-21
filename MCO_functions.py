
import matplotlib.pyplot as plt
import numpy as np # for importing datasets
import math
import pandas as pd
import warnings
from typing import Literal
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import torch
from torch import optim
import torch.nn as nn

# Used for binomial at least
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score, 
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def set_xtick_labels(ax, labels, rotation=45, ha="right"):
    """Set categorical x tick labels from a mapping dictionary."""
    ax.set_xticks([float(x) for x in range(0, len(labels))])
    ax.set_xticklabels(list(labels.values()), rotation=rotation, ha=ha)


def class_distribution_by_feature(df, col, label_map, class_label, urb_col="URB_LABEL"):
    """Return within-class percentage distribution for a categorical feature."""
    class_df = df[df[urb_col] == class_label]
    dist = class_df[col].value_counts(normalize=True).sort_values(ascending=True) * 100

    return pd.DataFrame(
        {
            "Label": [label_map.get(idx, str(idx)) for idx in dist.index],
            "Percent": dist.values,
        }
    )


def plot_single_class_distribution(
    ax,
    df,
    col,
    label_map,
    class_label,
    color,
    title,
    urb_col="URB_LABEL",
):
    """Plot a single class-specific horizontal percentage distribution chart."""
    plot_df = class_distribution_by_feature(df, col, label_map, class_label, urb_col=urb_col)
    y = np.arange(len(plot_df))

    ax.barh(y, plot_df["Percent"], color=color)

    for i, pct in enumerate(plot_df["Percent"]):
        if pct >= 2:
            ax.text(pct + 0.8, i, f"{pct:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Label"], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage within class (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


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

# confusion matrix for binomial
def class_accuracy_from_cm(cm):
    """Return rural and urban recall from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    rural_acc = tn / (tn + fp) if (tn + fp) else 0.0
    urban_acc = tp / (tp + fn) if (tp + fn) else 0.0
    return rural_acc, urban_acc

# metric chart for binomial
# showing difference with table column highlights
# compares metrics of model
def build_metric_series(y_true, y_pred, y_proba, cm, metric_order):
    """Build a metrics series in the exact order used by notebook comparison tables."""
    rural_acc, urban_acc = class_accuracy_from_cm(cm)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
        average=None,
    )
    prec = np.asarray(prec)
    rec = np.asarray(rec)
    f1 = np.asarray(f1)
    metric_map = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Rural Accuracy": rural_acc,
        "Urban Accuracy": urban_acc,
        "Rural Precision": float(prec[0]),
        "Rural Recall": float(rec[0]),
        "Rural F1": float(f1[0]),
        "Urban Precision": float(prec[1]),
        "Urban Recall": float(rec[1]),
        "Urban F1": float(f1[1]),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "Log Loss": log_loss(y_true, y_proba),
    }
    return pd.Series([metric_map[m] for m in metric_order], index=metric_order)

# shows all the metrics, including per-class metrics and confusion matrix, in a reusable way
def report_binary_metrics(
    y_true,
    y_pred,
    y_proba,
    title="Metrics",
    threshold=None,
    n_iter=None,
):
    """Print a reusable binary-classification report and return useful tables/values."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Rural (0)", "Actual Urban (1)"],
        columns=["Pred Rural (0)", "Pred Urban (1)"],
    )
    rural_acc, urban_acc = class_accuracy_from_cm(cm)

    print(f"{title} (Overall)")
    if threshold is not None:
        print(f"Threshold used  : {threshold:.2f}")
    print(f"Accuracy        : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Acc    : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC         : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR-AUC          : {average_precision_score(y_true, y_proba):.4f}")
    print(f"Log Loss        : {log_loss(y_true, y_proba):.4f}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
        average=None,
    )
    prec = np.asarray(prec)
    rec = np.asarray(rec)
    f1 = np.asarray(f1)

    class_metrics_df = pd.DataFrame(
        {
            "Rural (0)": [prec[0], rec[0], f1[0]],
            "Urban (1)": [prec[1], rec[1], f1[1]],
        },
        index=["Precision", "Recall", "F1-score"],
    )

    print("\nPer-Class Metrics")
    display(class_metrics_df.style.format("{:.4f}"))

    print("\nConfusion Matrix:")
    print(cm_df)
    if n_iter is not None:
        print("\nIterations used:", n_iter)

    return {
        "cm": cm,
        "cm_df": cm_df,
        "rural_acc": rural_acc,
        "urban_acc": urban_acc,
        "class_metrics_df": class_metrics_df,
    }

# Track training log loss across iterations of a logistic regression model (currently implemented in task 4 - training).
def track_logreg_loss_by_iteration(
    X_train,
    y_train,
    total_iterations,
    c=1.0,
    penalty: Literal["l1", "l2", "elasticnet"] | None = "l2",
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
    class_weight=None,
    random_state=42,
    head_n=10,
    tail_n=10,
    plot=True,
):

    if total_iterations <= 0:
        raise ValueError("total_iterations must be > 0")

    loss_tracker_model = LogisticRegression(
        penalty=penalty,
        C=c,
        solver=solver,
        class_weight=class_weight,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )

    iter_history = []
    for step in range(1, total_iterations + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            loss_tracker_model.fit(X_train, y_train)

        y_train_proba_step = loss_tracker_model.predict_proba(X_train)[:, 1]
        iter_history.append(
            {
                "Optimizer Iteration": step,
                "Train Log Loss": log_loss(y_train, y_train_proba_step),
            }
        )

    full_loss_df = pd.DataFrame(iter_history)

    compact_loss_df = pd.concat(
        [full_loss_df.head(head_n), full_loss_df.tail(tail_n)],
        axis=0,
    ).drop_duplicates(subset=["Optimizer Iteration"]).reset_index(drop=True)

    ax = None
    if plot:
        ax = full_loss_df.plot(
            x="Optimizer Iteration",
            y="Train Log Loss",
            marker="o",
            figsize=(8, 4),
            grid=True,
            legend=False,
            title=f"Training Log Loss over {total_iterations} iterations",
        )
        ax.set_ylabel("Log Loss")
        plt.show()

    return {
        "full_loss_df": full_loss_df,
        "compact_loss_df": compact_loss_df,
        "ax": ax,
    }

# to determine color (yellow, green red) just an aesthetic function
def highlight_val_train_row(row, direction_map, delta_col="Val - Train"):
    metric_name = row.name
    delta = row[delta_col]
    direction = direction_map.get(metric_name, "higher")

    if abs(delta) < 1e-12:
        color = "#be9100"  # tie
    else:
        improved = (delta > 0 and direction == "higher") or (delta < 0 and direction == "lower")
        color = "#007a1d" if improved else "#80000b"

    styles = [""] * len(row)
    if delta_col in row.index:
        delta_idx = row.index.get_loc(delta_col)
        styles[delta_idx] = f"background-color: {color}"
    return styles

# Tuning function that runs a grid search over C, solver, class weight, and decision threshold, and returns a dataframe of results sorted by balanced accuracy and gap.
def run_logreg_validation_grid(
    X_train,
    y_train,
    X_val,
    y_val,
    threshold_min=0.33,
    threshold_max=0.5,
    threshold_step=0.01,
):

    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")
    if threshold_min >= threshold_max:
        raise ValueError("threshold_min must be < threshold_max")

    thresholds = np.round(
        np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step),
        4,
    )
    thresholds = [float(t) for t in thresholds if 0.0 < t < 1.0]

    max_iter = 300
    param_grid = {
        # Wider log-scale range: strong regularization (small C) to weak regularization (large C).
        "C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0],
        "Solver": ["liblinear", "lbfgs"],
        "Class_Weight": [
            None,
            "balanced",
            {0: 1.0, 1: 1.05},
            {0: 1.0, 1: 1.1},
            {0: 1.0, 1: 1.15},
            {0: 1.0, 1: 1.2},
            {0: 1.05, 1: 1.0},
            {0: 1.1, 1: 1.0},
            {0: 1.15, 1: 1.0},
            {0: 1.2, 1: 1.0},
        ],
        "Threshold": thresholds,
    }

    rows = []
    for c in param_grid["C"]:
        for solver in param_grid["Solver"]:
            for cw in param_grid["Class_Weight"]:
                model = LogisticRegression(
                    penalty="l2",
                    C=c,
                    solver=solver,
                    class_weight=cw,
                    max_iter=max_iter,
                    random_state=42,
                )
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                for threshold in param_grid["Threshold"]:
                    val_pred = (val_proba >= threshold).astype(int)
                    cm = confusion_matrix(y_val, val_pred)
                    rural_acc, urban_acc = class_accuracy_from_cm(cm)
                    gap = abs(rural_acc - urban_acc)
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_val,
                        val_pred,
                        labels=[0, 1],
                        zero_division=0,
                        average=None,
                    )
                    prec = np.asarray(prec)
                    rec = np.asarray(rec)
                    f1 = np.asarray(f1)

                    rows.append(
                        {
                            "C": c,
                            "Solver": solver,
                            "Class_Weight": str(cw),
                            "Threshold": threshold,
                            "Accuracy": accuracy_score(y_val, val_pred),
                            "Balanced_Acc": balanced_accuracy_score(y_val, val_pred),
                            "Rural_Acc": rural_acc,
                            "Urban_Acc": urban_acc,
                            "Rural_Precision": float(prec[0]),
                            "Rural_Recall": float(rec[0]),
                            "Rural_F1": float(f1[0]),
                            "Urban_Precision": float(prec[1]),
                            "Urban_Recall": float(rec[1]),
                            "Urban_F1": float(f1[1]),
                            "Gap_For_Ranking": gap,
                            "ROC_AUC": roc_auc_score(y_val, val_proba),
                            "PR_AUC": average_precision_score(y_val, val_proba),
                            "LogLoss": log_loss(y_val, val_proba),
                        }
                    )

    candidate_df = pd.DataFrame(rows)
    candidate_df = candidate_df.sort_values(
        ["Balanced_Acc", "Gap_For_Ranking", "Accuracy"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    candidate_df = candidate_df.drop(columns=["Gap_For_Ranking"])
    return candidate_df

def trainNetwork(network, data_loader, 
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
        scores, probabilities = model.forward(X_val, verbose=False)
        predictions = model.predict(probabilities)

    # Since dataset is imbalanced, we use balanced accuracy instead of regular accuracy
    balanced_accuracy = balanced_accuracy_score(y_val, predictions)

    precision = precision_score(y_val, predictions, average=None)
    class_precisions = {f"class {i}": p for i, p in enumerate(precision)}

    recall = recall_score(y_val, predictions, average=None)
    class_recalls = {f"class {i}": r for i, r in enumerate(recall)}

    f1 = f1_score(y_val, predictions, average=None)
    class_f1s = {f"class {i}": f for i, f in enumerate(f1)}

    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, y_val)

    auc_score = roc_auc_score(y_val,probabilities[:,1])
    pr_auc = average_precision_score(y_val,probabilities[:,1])

    cm = confusion_matrix(y_val, predictions)

    return {
        "balanced_accuracy": balanced_accuracy,
        "precision": class_precisions,
        "recall": class_recalls,
        "f1": class_f1s,
        "log_loss": loss,
        "ROC_AUC": auc_score,
        "PR_AUC": pr_auc,
        "confusion_matrix": cm
    }

# This following lines contains the implementation of a feedforward neural network using PyTorch.
# These contents are based off the template given in the NeuralNetwork lab activity from DLSU's STINTSY course.

import torch.nn as nn
import torch.nn.init
from torch import optim

class NeuralNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 num_classes,
                 list_hidden,
                 activation='relu'):
        """Class constructor for NeuralNetwork

        Arguments:
            input_size {int} -- Number of features in the dataset
            num_classes {int} -- Number of classes in the dataset
            list_hidden {list} -- List of integers representing the number of
            units per hidden layer in the network
            activation {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.
        """
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation

    def create_network(self):
        """Creates the layers of the neural network.
        """
        layers = []

        # Append a torch.nn.Linear layer to the
        # layers list with correct values for parameters in_features and
        # out_features. This is the first layer of the network.
        # HINT: You will use self.list_hidden here.
        layers.append(torch.nn.Linear(in_features=self.input_size, out_features=self.list_hidden[0]))

        # Append the activation layer by calling
        # the self.get_activation() function.
        layers.append(self.get_activation(self.activation))

        # Iterate over other hidden layers just before the last layer
        for i in range(len(self.list_hidden) - 1):

            # Append a torch.nn.Linear layer to
            # the layers list according to the values in self.list_hidden.
            layers.append(torch.nn.Linear(in_features=self.list_hidden[i], out_features=self.list_hidden[i + 1]))

            # Append the activation layer by
            # calling the self.get_activation() function.
            layers.append(self.get_activation(self.activation))

        # Append a torch.nn.Linear layer to the
        # layers list with correct values for parameters in_features and
        # out_features. This is the last layer of the network.
        layers.append(torch.nn.Linear(in_features=self.list_hidden[-1], out_features=self.num_classes))
        
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        """Initializes the weights of the network. Weights of a
        torch.nn.Linear layer should be initialized from a normal
        distribution with mean 0 and standard deviation 0.1. Bias terms of a
        torch.nn.Linear layer should be initialized with a constant value of 0.
        """
        torch.manual_seed(2)

        # For each layer in the network
        for module in self.modules():

            # If it is a torch.nn.Linear layer
            if isinstance(module, nn.Linear):

                # Initialize the weights of the torch.nn.Linear layer
                # from a normal distribution with mean 0 and standard deviation
                # of 0.1.
                nn.init.normal_(module.weight, mean=0, std=0.1)

                # Initialize the bias terms of the torch.nn.Linear layer
                # with a constant value of 0.
                nn.init.constant_(module.bias, 0)

    def get_activation(self,
                       mode='sigmoid'):
        """Returns the torch.nn layer for the activation function.

        Arguments:
            mode {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.

        Returns:
            torch.nn -- torch.nn layer representing the activation function.
        """
        activation = nn.Sigmoid()

        if mode == 'tanh':
            activation = nn.Tanh()

        elif mode == 'relu':
            activation = nn.ReLU(inplace=True)

        return activation

    def forward_manual(self,
                       x,
                       verbose=False):
        """Forward propagation of the model, implemented manually.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
            features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
            output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
            the output of the final linear layer in the network. A Tensor of
            shape (N, C) representing the probabilities of each class given by
            the softmax function.
        """

        # For each layer in the network
        for i in range(len(self.layers) - 1):

            # If it is a torch.nn.Linear layer
            if isinstance(self.layers[i], nn.Linear):

                # Compute the result of the linear layer. Do not forget
                # to add the bias term. Assign the result to x.
                x = torch.matmul(x, self.layers[i].weight.t()) + self.layers[i].bias

            # If it is another function
            else:
                # Call the forward() function of the layer
                # and return the result to x.
                x = self.layers[i](x)

            if verbose:
                # Print the output of the layer
                print('Output of layer ' + str(i))
                print(x, '\n')

        # Apply the softmax function
        probabilities = self.layers[-1](x)

        if verbose:
            print('Output of layer ' + str(len(self.layers) - 1))
            print(probabilities, '\n')

        return x, probabilities

    def forward(self,
                x,
                verbose=False):
        """Forward propagation of the model, implemented using PyTorch.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
            features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
            output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
            the output of the final linear layer in the network. A Tensor of
            shape (N, C) representing the probabilities of each class given by
            the softmax function.
        """

        # For each layer in the network
        for i in range(len(self.layers) - 1):

            # Call the forward() function of the layer
            # and return the result to x.
            x = self.layers[i](x)

            if verbose:
                # Print the output of the layer
                print('Output of layer ' + str(i))
                print(x, '\n')

        # Apply the softmax function
        probabilities = self.layers[-1](x)

        if verbose:
            print('Output of layer ' + str(len(self.layers) - 1))
            print(probabilities, '\n')

        return x, probabilities

    def predict(self,
                probabilities):
        """Returns the index of the class with the highest probability.

        Arguments:
            probabilities {torch.Tensor} -- A Tensor of shape (N, C)
            representing the probabilities of N instances for C classes.

        Returns:
            torch.Tensor -- A Tensor of shape (N, ) contaning the indices of
            the class with the highest probability for N instances.
        """

        # Return the index of the class with the highest probability
        return torch.argmax(probabilities, dim=1)

# The following contents are based off the template given in the NeuralNetwork lab activity from DLSU's STINTSY course.

import numpy as np

class DataLoader(object):

    def __init__(self, X, y, batch_size):
        """Class constructor for DataLoader

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) containing the
            data; there are N samples each of dimension D.
            y {np.ndarray} -- A numpy array of shape (N, 1) containing the
            ground truth values.
            batch_size {int} -- An integer representing the number of instances
            per batch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.indices = np.array([i for i in range(self.X.shape[0])])
        np.random.seed(1)

    def shuffle(self):
        """Shuffles the indices in self.indices.
        """

        # Use np.random.shuffle() to shuffles the indices in self.indices
        np.random.shuffle(self.indices)

    def get_batch(self, mode='train'):
        """Returns self.X and self.y divided into different batches of size
        self.batch_size according to the shuffled self.indices.

        Arguments:
            mode {str} -- A string which determines the mode of the model. This
            can either be `train` or `test`.

        Returns:
            list, list -- List of np.ndarray containing the data divided into
            different batches of size self.batch_size; List of np.ndarray
            containing the ground truth labels divided into different batches
            of size self.batch_size
        """

        X_batch = []
        y_batch = []

        # If mode is set to `train`, shuffle the indices first using
        # self.shuffle().
        if mode == 'train':
            self.shuffle()
        elif mode == 'test':
            self.indices = np.array([i for i in range(self.X.shape[0])])

        # The loop that will iterate from 0 to the number of instances with
        # step equal to self.batch_size
        for i in range(0, len(self.indices), self.batch_size):

            # Check if we can still get self.batch_size from the
            # remaining indices starting from index i. Edit the condition
            # below.
            if i + self.batch_size <= len(self.indices):
                indices = self.indices[i:i + self.batch_size]

            # Else, just get the remaining indices from index i until the
            # last element in the list. Edit the statement inside the else
            # block.
            else:
                indices = self.indices[i:]

            X_batch.append(self.X[indices])
            y_batch.append(self.y[indices])

        return X_batch, y_batch
