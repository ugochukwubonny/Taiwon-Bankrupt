# Taiwon-Bankrupt

## Project Overview
This project aims to tackle the issue of imbalanced data in the Taiwon Bankrupt dataset. Using various machine learning techniques, we will explore the features of the dataset, utilize visualizations for better understanding, and develop models to handle the imbalance through under-sampling and over-sampling. Additionally, we will expand our analysis to include ensemble models such as Random Forest and Gradient Boosting Trees to improve prediction accuracy.

## Models Implemented
Decision Tree Classifier with Under- and Over-Sampling
Description: This model employs a Decision Tree Classifier and addresses the imbalance in the dataset by using both Random Under-Sampling and Random Over-Sampling techniques.

## Libraries Used:

json
pandas
matplotlib
plotly.express
seaborn
pickle
imblearn.over_sampling.RandomOverSampler
imblearn.under_sampling.RandomUnderSampler
sklearn.metrics.ConfusionMatrixDisplay
sklearn.model_selection.train_test_split
sklearn.impute.SimpleImputer
sklearn.pipeline.Pipeline
sklearn.tree.DecisionTreeClassifier

## Performance:
Baseline Accuracy: 0.97
Training and Test Accuracy:
Regular Model: 1.0 (train), 0.9523 (test)
Under-Sampling: 0.8203 (train), 0.8101 (test)
Over-Sampling: 1.0 (train), 0.9531 (test)

## Ensemble Models: Random Forest Classifier
Description: This model expands the Decision Tree into a Random Forest, an ensemble model that builds multiple decision trees and merges them together to get a more accurate and stable prediction. It also includes hyperparameter tuning using GridSearchCV.

## Libraries Used:

json
pickle
pandas
matplotlib
imblearn.over_sampling.RandomOverSampler
sklearn.impute.SimpleImputer
sklearn.model_selection.GridSearchCV
sklearn.model_selection.cross_val_score
sklearn.model_selection.train_test_split
sklearn.ensemble.RandomForestClassifier
sklearn.pipeline.Pipeline
sklearn.metrics.ConfusionMatrixDisplay
Performance:

Baseline Accuracy: 0.97
Training Accuracy: 1.0
Test Accuracy: 0.967

## Gradient Boosting Trees
Description: This model uses Gradient Boosting, another ensemble method that builds models sequentially with each new model correcting errors made by the previous ones. It is particularly effective for imbalanced datasets.

## Libraries Used:

json
pandas
matplotlib
ipywidgets
pickle
imblearn.over_sampling.RandomOverSampler
sklearn.metrics.ConfusionMatrixDisplay
sklearn.metrics.classification_report
sklearn.metrics.confusion_matrix
sklearn.impute.SimpleImputer
sklearn.model_selection.GridSearchCV
sklearn.model_selection.train_test_split
sklearn.ensemble.GradientBoostingClassifier
ipywidgets.interact
sklearn.pipeline.Pipeline
Performance:

Baseline Accuracy: 0.97
Training Accuracy: 1.0
Test Accuracy: 0.92
Data Source
The dataset contains financial data of companies in Taiwon and indicates whether they went bankrupt. The data includes various financial attributes that help in predicting the bankruptcy status.

## Methodology
Data Exploration: Loaded and explored the dataset using visualizations to understand the distribution of features and identify the imbalance.
Data Preprocessing: Cleaned and imputed missing values to prepare the data for modeling.

## Model Building:
Implemented Decision Tree Classifier with under- and over-sampling techniques.
Developed a Random Forest model with hyperparameter tuning using GridSearchCV.
Built a Gradient Boosting model to handle imbalanced data effectively.
Model Evaluation: Evaluated the models based on training and test accuracy, using metrics like confusion matrix and classification report.
Visualization: Used matplotlib and plotly to visualize the results and model performance.

## Results
The Decision Tree with over-sampling provided a slight improvement over the baseline.
The Random Forest model outperformed the Decision Tree models, achieving the highest test accuracy.
The Gradient Boosting model, while slightly lower in test accuracy than Random Forest, provided robust predictions and handled the imbalance well.

## Conclusion
This project demonstrates the application of various machine learning techniques to handle imbalanced data and improve prediction accuracy. The ensemble models, particularly Random Forest and Gradient Boosting, proved to be effective in predicting bankruptcy in Taiwon companies.
