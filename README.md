# 🩺 Breast Cancer Prediction Using Artificial Neural Networks (ANN)

## Project Overview

This repository contains a project dedicated to the prediction of breast cancer diagnosis (malignant or benign) by leveraging an Artificial Neural Network (ANN). The model is constructed and trained using the well established Breast Cancer Wisconsin Dataset.

The primary objective is to develop a robust machine learning pipeline, encompassing data preprocessing, exploratory data analysis, model construction, and rigorous performance evaluation.

## Dataset

The data originates from the [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) available on Kaggle.

* **Total Samples**: 569
* **Features**: 32 (numeric measurements from digitized images of breast masses)
* **Target**: `diagnosis` — M (malignant) / B (benign)

## Project Workflow

The notebook follows a comprehensive workflow for model development and analysis:

1.  **Data Loading & Preparation**: The dataset is retrieved using the `kagglehub` library and loaded into a `pandas` DataFrame.
2.  **Data Cleaning**: Unnecessary columns, such as 'id' and 'Unnamed: 32', are identified and removed to streamline the dataset.
3.  **Exploratory Data Analysis (EDA)**: The distribution of benign versus malignant diagnoses is visualized using count plots and pie charts. A correlation heatmap is generated to identify the features most strongly correlated with the diagnosis.
4.  **Preprocessing**:
    * The categorical target variable (`diagnosis`) is numerically encoded (M=1, B=0).
    * All features are standardized using `StandardScaler` from scikit learn to ensure the model treats all inputs equitably.
5.  **Train Test Split**: The data is partitioned into training and testing sets using `train_test_split`.
6.  **Model Architecture**: A Sequential Artificial Neural Network is constructed using TensorFlow (Keras). The architecture includes an `InputLayer`, multiple `Dense` layers, and `Dropout` layers to prevent overfitting.
7.  **Model Training**: The ANN is compiled using an appropriate optimizer and loss function (e.g., 'binary_crossentropy') and trained on the training data.
8.  **Evaluation**: The model's predictive performance is assessed on the unseen test set. Results are presented using a `classification_report` (detailing precision, recall, and f1 score) and a `confusion_matrix`.
9.  **Model Explainability**: The project employs SHAP (SHapley Additive exPlanations) to interpret the model's predictions. A SHAP `force_plot` is generated to illustrate which features contribute most significantly to the diagnosis for a specific instance.

## Dependencies

This project utilizes the following core Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* plotly
* scikit-learn (sklearn)
* tensorflow (keras)
* kagglehub
* shap

To install the necessary packages, you can run:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn tensorflow kagglehub shap