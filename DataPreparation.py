"""
ILPD (Indian Liver Patient Dataset) Data Set
http://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29
Direct Link: http://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv
"""

import os.path
import pandas as pd

DATA_FILE_PATH = "Indian Liver Patient Dataset (ILPD).csv"
CLEAN_DATA_FILE_PATH = "Cleaned_Indian Liver Patient Dataset (ILPD).csv"
_df = None


def get_clean_data():

    if _df is None:
        _construct_clean_data()

    # Dangerous here! We should return a copy so that the original data is undisturbed.
    return _df.copy()


def _construct_clean_data():
    global _df
    if os.path.isfile(CLEAN_DATA_FILE_PATH):
        # Load the already cleansed data.
        _df = pd.read_csv(CLEAN_DATA_FILE_PATH)
    else:
        if os.path.isfile(DATA_FILE_PATH):
            temp_df = pd.read_csv(DATA_FILE_PATH, header=None)

            # Check numbers of rows and columns.
            (num_row, num_col) = temp_df.shape
            if num_row != 583 or num_col != 11:
                raise Exception("Dataset tampered")

            # Add meaningful columns' names.
            temp_df.columns = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
                               'alamine_aminotransferase',
                               'aspartate_aminotransferase', 'total_protiens', 'albumin',
                               'ratio_albumin_and_globulin_ratio', 'liver_res']

            # Transform categorical target variable to numberic one. 1 means "positive", 0 means "negative".
            temp_df['liver_res'] = temp_df['liver_res'].apply(lambda x: 0 if (x == 2) else x)

            # Column "ratio_albumin_and_globulin_ratio" has some NULL values.
            index_null = temp_df['ratio_albumin_and_globulin_ratio'].index[temp_df['ratio_albumin_and_globulin_ratio'].isnull()]
            # 209, 241, 253, 312 ==> Remove these rows!
            # Even more, with NaN, seaborn library has problems!
            temp_df = temp_df.drop(temp_df.index[index_null])

            # Save clean data to file.
            temp_df.to_csv(CLEAN_DATA_FILE_PATH, index=False)

            _df = temp_df
        else:
            raise IOError("%s not found" % DATA_FILE_PATH)
