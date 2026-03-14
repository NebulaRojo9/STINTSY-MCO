
import matplotlib.pyplot as plt
import numpy as np # for importing datasets
import math
import pandas as pd


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