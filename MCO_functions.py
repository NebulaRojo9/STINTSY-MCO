
import matplotlib.pyplot as plt
import numpy as np # for importing datasets
import math
import pandas as pd
import warnings
from typing import Literal
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# Used for binomial at least
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


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
    metric_map = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Rural Accuracy": rural_acc,
        "Urban Accuracy": urban_acc,
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
            "Rural (0)": [rural_acc, prec[0], rec[0], f1[0]],
            "Urban (1)": [urban_acc, prec[1], rec[1], f1[1]],
        },
        index=["Accuracy", "Precision", "Recall", "F1-score"],
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


def highlight_val_train_row(row, direction_map, delta_col="Val - Train"):
    """Color only the delta cell by whether validation moved in the preferred direction."""
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


def run_logreg_validation_grid(
    X_train,
    y_train,
    X_val,
    y_val,
    threshold_min=0.33,
    threshold_max=0.5,
    threshold_step=0.01,
):
    """Evaluate candidate logistic-regression settings and rank by validation quality.

    Thresholds are generated from threshold_min to threshold_max (inclusive)
    using threshold_step so the search can cover finer decision cutoffs.
    """
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