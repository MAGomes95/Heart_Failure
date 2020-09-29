import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC


def augment_dataset(
        data: pd.DataFrame,
        categorical_features: list,
        target_feature: str = 'target',
        k_parameter: int = 3
) -> pd.DataFrame:
    """Over-Sampling dataset augmentation

    The main function will augment the given dataset
    with synthetic instances associated to the less
    representative target class, through a variant of
    SMOTE that allows categorical features

    Args:
        data: Dataset to resample
        categorical_features: Names of the categorical features
        target_feature: Name of the target class. By default, 'target'
        k_parameter: Number of nearest neighbours to be used for the synthetic model

    Returns:
        Augmented dataset
    """

    categorical_mask = [True if feature in categorical_features else False for feature in data.columns]

    smote_obj = SMOTENC(
        categorical_features=categorical_mask,
        sampling_strategy='minority',
        k_neighbors=k_parameter
    )

    features_resampled, target_resampled = smote_obj.fit_sample(
        data[np.setdiff1d(data.columns, target_feature)], data[target_feature]
    )

    resampled_data = pd.DataFrame(
        data=features_resampled,
        columns=np.setdiff1d(data.columns, [target_feature])
    )

    resampled_data[target_feature] = target_resampled

    return resampled_data

