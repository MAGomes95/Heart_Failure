import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def train_validation_test_split(
        data: pd.DataFrame,
        target: str,
        val_partition: float = 0.2,
        test_partition: float = 0.15
) -> list:
    """The functions aims to partion the data into three subsets

    The main function attempts to partition the data as a function
    of the percentages given as argument, in a stratified manner according
    with the target feature

    Args:
        data: Dataset, from which the subsets will be extracted
        target: Name of the target
        val_partition: Percentage of the dataset to extract for validation
        test_partition: Percentage of the dataset to extract for test

    Returns:
        Three subsests of data, according with the percentages given
    """

    assert val_partition + test_partition < 1.0

    val_samples = val_partition * data.shape[0]
    test_samples = test_partition * data.shape[0]

    train_validation, test = train_test_split(
        data, test_size=int(test_samples), stratify=data[target]
    )

    train, validation = train_test_split(
        train_validation, test_size=int(val_samples), stratify=train_validation[target]
    )

    return [train, validation, test]


def hyperparameter_tunning(
        model,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        hyperparameter_grid: dict
) -> dict:
    """Hyperparameter tunning process

    The current function aims to identify the best set
    os hyperparameters among the hyperparameter_grid. The
    model will then be trained in the training set and evaluated
    in the validation set.

    Args:
        model: XGBoost, CatBoost, LightGBM or Scikit-learn model
        train_features: Training feature subspace
        train_target: Training target vector
        validation_features: Validation feature subspace
        validation_target: Validation target vector
        hyperparameter_grid: Hyperparameter range

    Returns:
        The best hyperparameter subset, fitted model and its metric (f1-score)
    """

    best_estimator = None
    best_hyperparams = {}
    best_metric = 0.0

    hp_grid = [this_hp for this_hp in hyperparameter_grid.values()]
    all_combinations_list = list(itertools.product(*hp_grid))

    all_combinations_dic = []

    for this_combination in all_combinations_list:

        this_hp_set = {}

        for i, key in enumerate(hyperparameter_grid.keys()):

            this_hp_set[key] = this_combination[i]

        all_combinations_dic.append(this_hp_set)

    for this_hp_set in all_combinations_dic:

        this_estimator = model(**this_hp_set)

        this_estimator.fit(train_features, train_target)

        predictions = this_estimator.predict(validation_features)

        evaluation_metric = f1_score(validation_target, predictions)

        if evaluation_metric > best_metric:

            best_metric = evaluation_metric

            best_estimator = this_estimator

            best_hyperparams = this_hp_set

    return {'best_hyperparameters': best_hyperparams, 'best_model': best_estimator, 'best_metric': best_metric}

