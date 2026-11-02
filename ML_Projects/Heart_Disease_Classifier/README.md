# 🫀 Predicting Heart Disease with Machine Learning

## Introduction

Heart disease is one of the leading causes of death worldwide, and early detection can save lives. Doctors often decide whether a patient needs further tests by looking at clinical measurements such as age, blood pressure, cholesterol, heart rate and chest pain symptoms.

In this project I build a **binary classification model** that predicts whether a patient has heart disease (`1`) or not (`0`), using 13 clinical features from the UCI Heart Disease dataset (303 patients, reduced to 296 after cleaning).

### What this notebook covers

| Step | What I did |
|:--|:--|
| **1. Exploratory data analysis** | Checked for missing values and duplicates, cleaned invalid category codes, and studied the relationships between features and the target. |
| **2. Preprocessing** | One-hot encoded the categorical features and scaled the numerical features. |
| **3. Model building** | Trained a Random Forest classifier as a baseline. |
| **4. Improving the model** | Tuned hyperparameters with `RandomizedSearchCV`. |
| **5. Evaluation** | Measured accuracy, precision, recall, F1-score, ROC curve and confusion matrix on a held-out test set. |
| **6. Saving the model** | Exported the trained model with `pickle` so it can be reused. |

### Goal

Reach an accuracy of **over 85 %** on the test set.

### Summary of results

The tuned Random Forest reached **86.67 % accuracy** on the test set, with a precision of 0.90 and recall of 0.84 for detecting heart disease. This meets the goal, but the test set has only 60 patients, so the result is an estimate rather than a guarantee. The detailed analysis, limitations and future work are in the **Conclusion** section at the end of the notebook.

> ⚠️ This project is for learning purposes only.

