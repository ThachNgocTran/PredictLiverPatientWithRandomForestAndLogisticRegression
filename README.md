# Predict Liver Patient With Random Forest And Logistic Regression

Following are the highlights, hopefully some can capture your attention!

* The dataset ILPD (Indian Liver Patient Dataset) [1]  comprises 583 instances with each having 10 features and 1 target variable. The dataset is used to classify if a patient, given a feature vector, has the Liver Disease or not (binary classification).
* The predictive algorithms Random Forest and Logistic Regression are chosen for this task.
* The task is made possible thanks to Python, and especially Scikit-Learn/Pandas libraries. Indeed, I used Anaconda3 [2] for “all-in-one” installation.
* GridSearchCV is used to automatically search for optimal parameters in Random Forest and Logistic Regression.
* I missed ggplot2 in R, but in Python for Data Science, seaborn [3] seems promising. This library is used to make Feature Scatter plot.
* The algorithms’ performance is compared using ROC (Receiver operating characteristic) and AUC (Area Under Curve).
* The quickest way to reproduce this report is to run “python.exe ClassificationTask.py” with “DataPreparation.py” and “Indian Liver Patient Dataset (ILPD).csv” put in the same place.

Please see my article:

* https://thachtranerc.wordpress.com/2016/05/15/predict-liver-disease-with-random-forest-and-logistic-regression/

## Software Environment:

* Anaconda3 v4.0.0 64bit (Python v3.5.2)
* PyCharm 2016.1.3
* IPython Notebook

## Reference:

1. [ILPD (Indian Liver Patient Dataset) Data Set](http://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29)

2. [Anaconda3](https://www.continuum.io/downloads)

3. [Seaborn: statistical data visualization](https://stanford.edu/~mwaskom/software/seaborn/)
