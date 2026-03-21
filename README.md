# Predicting Urban vs. Rural Household Classification using Supervised Machine Learning

**Course:** Advanced Intelligent Systems (STINTSY)  
**Term:** Term 2 A.Y. 2025-2026  
**Group:** Group #2 CSINTSY Survivors (Section S18)  
**Professor:** Arren Matthew Capuchino Antioquia  

## Project Overview
This repository contains the end-to-end machine learning pipeline for classifying Philippine households as either Urban or Rural based on the Family Income and Expenditure Survey (FIES) dataset. The project compares four distinct supervised machine learning algorithms to determine the most robust predictive model through rigorous hyperparameter tuning and performance evaluation.


## Repository Structure

The project is organized into modular directories and files to prevent data leakage and ensure consistency across all models.

### Directories
* **`/feis_dataset/`**: Contains the raw 2012 FIES Public Use Files (CSV, data dictionaries, and reference metadata).
* **`/processed_data/`**: Stores the preprocessed, feature-selected, and scaled datasets (Train, Validation, Test splits) exported as `.pkl` files. This ensures every model trains on the exact same data points.
* **`/model_outputs/`**: Contains the final exported artifacts from each model notebook. This includes the final post-tuned models (`.pkl`) and their respective predictions on the test set (`.csv`).

### Python Scripts
* **`MCO_functions.py`**: A centralized script containing custom helper functions used across multiple notebooks.

### Jupyter Notebooks
* **`Preliminaries.ipynb`**: The foundational notebook handling Data Loading, Exploratory Data Analysis (EDA), and Data Preprocessing.
* **`Binomial Logistic Regression.ipynb`**: Implementation, tuning, and evaluation of the Logistic Regression model.
* **`Random Forest Classifier.ipynb`**: Implementation, tuning, and evaluation of the Random Forest model.
* **`Gradient Boosting Classifier.ipynb`**: Implementation, tuning, and evaluation of the Gradient Boosting model.
* **`Multi-Layer Perceptron.ipynb`**: Implementation, tuning, and evaluation of the Neural Network (MLP) model.
* **`Model Comparison.ipynb`**: The notebook that loads all exported test predictions and evaluates the four models side-by-side to declare the best one.


## Order of Execution

To reproduce the results of this project, the Jupyter Notebooks **must** be executed in the following order:

### Phase 1: Data Preparation
1. **`Preliminaries.ipynb`**


### Phase 2: Model Training & Tuning
The following four notebooks can be run in **any order** or simultaneously. They act independently but rely on the `processed_data` generated in Phase 1. 

2. **`Binomial Logistic Regression.ipynb`**
3. **`Random Forest Classifier.ipynb`**
4. **`Gradient Boosting Classifier.ipynb`**
5. **`Multi-Layer Perceptron.ipynb`**

### Phase 3: Final Evaluation
6. **`Model Comparison.ipynb`**


## Required Libraries
To successfully run this repository, ensure you have the following Python packages installed:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `joblib`