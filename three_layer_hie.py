"""
Author: Alex (Tai-Jung) Chen

Implement the 3-level hierarchical method.
"""
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier

N_REP = 1


def three_layer_hie(model_type: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                  y_test: pd.DataFrame) -> pd.DataFrame:

    '''data preprocessing'''
    # read data
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    # df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
    df.drop(['PCA1', 'PCA2'], axis=1, inplace=True)

    # data preprocessing
    # df = pd.get_dummies(df, columns=['type'], dtype=int)

    # reorder columns
    # all_columns = list(df.columns)
    # last_columns = all_columns[-3:]
    # remaining_columns = all_columns[:-3]
    # new_order = last_columns + remaining_columns
    # df_final = df[new_order]
    df_final = df

    # turn into np array
    X = df_final.iloc[:, :-2]#.to_numpy()
    y = df_final.iloc[:, -1]#.astype(int).to_numpy()

    '''modeling seq'''
    for rand_state in range(N_REP):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=rand_state)

        # binary model for normal vs. failure
        y_train_bin = y_train.apply(lambda x: 1 if x != 0 else 0)
        bin_model = xgb.XGBClassifier().fit(X_train, y_train_bin)

        # multi-class model for failure types
        multi_model = xgb.XGBClassifier().fit(X_train[y_train != 0], y_train[y_train != 0] - 1)

        # local models
        local_models = {}
        for sub in range(1, len(df_final['failure.type'].value_counts())):
            model = xgb.XGBClassifier()

            # select only majority and minority sub
            X_train_local = X_train[(y_train == sub) | (y_train == 0)]
            y_train_local = y_train[(y_train == sub) | (y_train == 0)]
            y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1
            model.fit(X_train_local, y_train_local)

            local_models[f"local_model_{sub}"] = model

        # testing
        y_pred_top = bin_model.predict(X_test)
        y_pred_sub = multi_model.predict(X_test[y_pred_top == 1]) + 1

        y_pred_final = np.full((len(y_test), ), 0)
        for sub in range(1, len(df_final['failure.type'].value_counts())):
            mdl = f"local_model_{sub}"
            indices = np.where(y_pred_sub == sub)
            y_pred_final[indices[0]] = local_models[mdl].predict(X_test.iloc[indices[0], :])

        # collapsed y_test to binary
        y_test = np.where(y_test != 0, 1, 0)

        # classification performance
        print("Proposed model performance")
        print(confusion_matrix(y_test, y_pred_final))
        print(classification_report(y_test, y_pred_final))

