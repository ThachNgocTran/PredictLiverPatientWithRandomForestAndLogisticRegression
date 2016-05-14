import os.path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

import DataPreparation

FEATURE_CORR_FILE_PATH = "featureCorrelation.png"
AUC_FILE_PATH = "AUCRFLR.png"
CROSS_VALIDATION_FOR_GRIDSEARCH = 5
RATIO_TRAINING_TESTING = 0.7

# For reproducibility.
RANDOM_SEED = 12345

# ******************* Preparing Data *******************
df = DataPreparation.get_clean_data()

# Hint for PyCharm to recognize it as "Dataframe", so autocompletion feature.
assert isinstance(df, pd.DataFrame)

# ******************* Exploratory Data Analysis *******************
drawn_df = df.copy()

# seaborn requires "hue" column to be of non-number.
# All other columns must be of number.
if not os.path.isfile(FEATURE_CORR_FILE_PATH):
    drawn_df['gender'] = drawn_df['gender'].apply(lambda x: 1 if (x == "Male") else 0)
    drawn_df['liver_res'] = drawn_df['liver_res'].apply(lambda x: "LIVER" if (x == 1) else "NONE")

    sns_plot = sns.pairplot(drawn_df, hue="liver_res")
    sns_plot.savefig(FEATURE_CORR_FILE_PATH)
    plt.close()

"""
Looking at the graph output.png, we can see that there are correlations between "direct_bilirubin" and "total_bilirubin"
(strong), "aspartate_aminotransferase" and "alamine_aminotransferase" (a little strong), "albumin" and "total_protiens"
(a little strong), "ratio_albumin_and_globulin_ratio" and "albumin" (a little strong).

Let's calculate the Standard correlation coefficient (pearson)!
"""

"""
drawn_df.dtypes:
age                                   int64
gender                                int64
total_bilirubin                     float64
direct_bilirubin                    float64
alkaline_phosphotase                  int64
alamine_aminotransferase              int64
aspartate_aminotransferase            int64
total_protiens                      float64
albumin                             float64
ratio_albumin_and_globulin_ratio    float64
liver_res                            object
dtype: object
"""

print("Correlation between features:\n%s\n" % str(drawn_df.corr()))

"""
Corr("direct_bilirubin" and "total_bilirubin") = 0.874481 => very strong
Corr("aspartate_aminotransferase" and "alamine_aminotransferase") = 0.791862 ==> strong
Corr("albumin" and "total_protiens") = 0.783112 ==> strong
Corr("ratio_albumin_and_globulin_ratio" and "albumin") = 0.689632 ==> quite strong
Corr("total_protiens" and "ratio_albumin_and_globulin_ratio") = 0.234887 ==> weak

So it makes sense to drop some columns of these to help not distort the model.
Columns deleted: direct_bilirubin (in favor of "total" sense), aspartate_aminotransferase (random choice),
                    albumin (to keep more columns (total_protiens, ratio_albumin_and_globulin_ratio)).
"""

print("Dropping column: direct_bilirubin, aspartate_aminotransferase, albumin.")
df.drop(['direct_bilirubin', 'aspartate_aminotransferase', 'albumin'], axis=1, inplace=True)

# We should have 7 features and 1 target.
df['gender'] = df['gender'].apply(lambda x: 1 if (x == "Male") else 0)

"""
df.dtypes:
age                                   int64
gender                                int64
total_bilirubin                     float64
alkaline_phosphotase                  int64
alamine_aminotransferase              int64
total_protiens                      float64
ratio_albumin_and_globulin_ratio    float64
liver_res                             int64
dtype: object

df.shape:
(579, 8)
"""

# ******************* Split the data into Training and Testing *******************
np.random.seed(RANDOM_SEED)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= RATIO_TRAINING_TESTING
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

features = df.columns[:7]       # Extract feature names from 0 to 7
target = df.columns[7]          # The target

# ******************* Random Forest *******************
# Search for optimal parameters.
param_grid_rf = {
    'n_estimators': np.arange(5, 20, 1).tolist(),
    'max_features': np.arange(1, 7, 1).tolist(),
    'criterion': ["entropy", "gini"]
}

CV_clfRf = GridSearchCV(estimator=RandomForestClassifier(n_jobs=2, random_state=RANDOM_SEED),
                        param_grid=param_grid_rf,
                        cv=CROSS_VALIDATION_FOR_GRIDSEARCH,
                        n_jobs=1,
                        verbose=1)

# 900 fits ==> 3.2min ==> 0.213s/fit
CV_clfRf.fit(train[features], train[target])
print("Best params for Random Forest: %s\n" % str(CV_clfRf.best_params_))

# {'criterion': 'entropy', 'max_features': 5, 'n_estimators': 15}

# Now construct the main classifier.
clfRf = RandomForestClassifier(n_jobs=2, random_state=RANDOM_SEED,
                               criterion="entropy",
                               max_features=5,
                               n_estimators=15)

# Use the training data to build the model.
clfRf.fit(train[features], train[target])

# Predict the test dataset.
preds_rf = clfRf.predict(test[features])
print("Prediction Result for Random Forest:\n%s\n" % str(pd.crosstab(preds_rf, test[target], rownames=['preds'], colnames=['actual'])))

# Get the probability rate with the test dataset.
preds_proba_rf = clfRf.predict_proba(test[features])[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(test[target], preds_proba_rf)


# ******************* Linear Regression *******************
# Search for optimal parameters.
param_grid_lr = {
    'C': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'penalty': ['l2', 'l1'],
    'fit_intercept': [True, False]
}

CV_clfLr = GridSearchCV(estimator=LogisticRegression(n_jobs=2, random_state=RANDOM_SEED),
                        param_grid=param_grid_lr,
                        cv=CROSS_VALIDATION_FOR_GRIDSEARCH,
                        n_jobs=1,
                        verbose=1)

# 220 fits => 0.8s ==> 0.00364s/fit
CV_clfLr.fit(train[features], train[target])
print("Best params for Logistic Regression: %s\n" % str(CV_clfLr.best_params_))

# {'C': 1.0, 'fit_intercept': True, 'penalty': 'l1'}
clfLr = LogisticRegression(n_jobs=2, random_state=RANDOM_SEED,
                                C=1,
                                fit_intercept=True,
                                penalty='l1')

# Use the training data to build the model.
clfLr.fit(train[features], train[target])

# Predict the test dataset.
preds_lr = clfLr.predict(test[features])
print("Prediction Result for Logistic Regression:\n%s\n" % str(pd.crosstab(preds_lr, test[target], rownames=['preds'], colnames=['actual'])))

# Get the probability rate with the test dataset.
preds_proba_lr = clfLr.predict_proba(test[features])[:, 1]
fpr_lr, tpr_lr, _ = metrics.roc_curve(test[target], preds_proba_lr)

# ******************* Evaluation *******************
# Draw the ROC to compare those algorithms.
if not os.path.isfile(AUC_FILE_PATH):
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], 'k-', label='Random Guess')
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.legend(loc='upper left', fontsize=12)

    plt.text(0.6, 0.2, 'AUC RF: %s' % (round(metrics.auc(fpr_rf, tpr_rf), 3)), fontsize=20)
    plt.text(0.6, 0.1, 'AUC LR: %s' % (round(metrics.auc(fpr_lr, tpr_lr), 3)), fontsize=20)

    plt.savefig(AUC_FILE_PATH)
    plt.close()

# ******************* Conclusion *******************
