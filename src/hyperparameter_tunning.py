from joblib import dump

import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.features.utils import hyperparameter_tunning


if __name__ == '__main__':

    # Data Reading

    data = {}

    # Linear Based Data

    train = pd.read_csv(
        '../data/silver/linear_based/train.csv',
        index_col=0
    )

    train_features = train[np.setdiff1d(train.columns, 'target')]
    train_target = train['target']

    validation = pd.read_csv(
        '../data/silver/linear_based/validation.csv',
        index_col=0
    )

    validation_features = validation[np.setdiff1d(validation.columns, 'target')]
    validation_target = validation['target']

    data['linear_based'] = {
        'train_features': train_features,
        'train_target': train_target,
        'validation_features': validation_features,
        'validation_target': validation_target
    }

    # Tree Based Data

    train = pd.read_csv(
        '../data/silver/tree_based/train.csv',
        index_col=0
    )

    train_features = train[np.setdiff1d(train.columns, 'target')]
    train_target = train['target']

    validation = pd.read_csv(
        '../data/silver/tree_based/validation.csv',
        index_col=0
    )

    validation_features = validation[np.setdiff1d(validation.columns, 'target')]
    validation_target = validation['target']

    data['tree_based'] = {
        'train_features': train_features,
        'train_target': train_target,
        'validation_features': validation_features,
        'validation_target': validation_target
    }

    categorical_features = [
        feature for feature in train_features.columns if train[feature].nunique() == 2
    ]

    # Model Definition

    models = {
        'lr': [LogisticRegression, 'linear_based'],
        'svm': [svm.SVC, 'linear_based'],
        'rf': [RandomForestClassifier, 'tree_based'],
        'xgb': [xgb.XGBClassifier, 'tree_based'],
        'cb': [cb.CatBoostClassifier, 'tree_based'],
        'gb': [GradientBoostingClassifier, 'tree_based']
    }

    hyperparameters = {
        'lr': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 0.5, 1.0, 2.0],
            'solver': ['liblinear'],
            'class_weight': ['balanced'],
            'max_iter': [200],
        },
        'svm': {
            'C': [0.1, 0.5, 1.0, 2.0],
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 5],
            'class_weight': ['balanced'],
        },
        'rf': {
            'n_estimators': [20, 40, 80, 100, 150],
            'max_depth': [2, 3, 5],
            'min_samples_leaf': [0.05, 0.1, 0.2],
            'criterion': ['gini', 'entropy'],
        },
        'xgb': {
            'max_depth': [2, 3, 5, 7, 10],
            'min_child_weight': [5, 7, 10],
            'subsample': [1.0, 0.9],
            'eta': [0.01, 0.05, 0.1, 0.2],
        },
        'cb': {
            'iterations': [20, 40, 80, 100, 150, 200],
            'depth': [2, 3, 5],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'random_strength': [0.1, 0.5, 1.0, 2.0],
            'bagging_temperature': [0, 1],
            'l2_leaf_reg': [0.2, 1, 2, 3.5],
            'cat_features': [categorical_features],
            'verbose': [False]
        },
        'gb': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [40, 80, 100, 125],
            'subsample': [0.8, 0.9, 1.0],
            'max_depth': [2, 3, 5],
            'min_samples_split': [2, 3, 5],
        }
    }

    # Hyperparameter Tunning and Model saving

    for this_model in models.items():

        model_name = this_model[0]

        model_object = this_model[1][0]

        model_type = this_model[1][1]

        this_output = hyperparameter_tunning(
            model=model_object,
            train_features=data[model_type]['train_features'],
            train_target=data[model_type]['train_target'],
            validation_features=data[model_type]['validation_features'],
            validation_target=data[model_type]['validation_target'],
            hyperparameter_grid=hyperparameters[model_name]
        )

        dump(
            value=this_output['best_model'],
            filename=f'../models/trained/{model_name}.joblib'
        )
