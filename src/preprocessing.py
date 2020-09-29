import numpy as np
import pandas as pd

from src.features.utils import train_validation_test_split
from src.features.preprocessing import apply_interaction_generator
from src.features.preprocessing import apply_scaler
from src.features.preprocessing import apply_pca
from src.features.augmention import augment_dataset
from src.features.clustering import membership_feature

if __name__ == '__main__':

    # Data Loading

    data = pd.read_csv(
        '../data/silver/expanded_dataset.csv',
        index_col=0
    )

    train, validation, test = train_validation_test_split(
        data=data,
        target='target'
    )

    # Data Preprocessing

    categorical_features = [
        feature for feature in data.columns if data[feature].nunique() == 2
    ]

    train = augment_dataset(
        data=train,
        categorical_features=categorical_features
    )

    tree_train, tree_validation, tree_test = apply_interaction_generator(
        fit_subset=train,
        transform_subsets=[validation, test],
        features=np.setdiff1d(train.columns, categorical_features)
    )

    continuous_features = np.setdiff1d(tree_train.columns, categorical_features)

    train, validation, test = apply_scaler(
        fit_subset=tree_train,
        transform_subset=[tree_validation, tree_test],
        features=continuous_features
    )

    train, validation, test = apply_pca(
        fit_subset=train,
        transform_subset=[validation, test],
        features=continuous_features
    )

    # Data Saving

    train.to_csv('../data/silver/linear_based/train.csv')
    validation.to_csv('../data/silver/linear_based/validation.csv')
    test.to_csv('../data/silver/linear_based/test.csv')

    tree_train.to_csv('../data/silver/tree_based/train.csv')
    tree_validation.to_csv('../data/silver/tree_based/validation.csv')
    tree_test.to_csv('../data/silver/tree_based/test.csv')
