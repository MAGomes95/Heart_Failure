import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


def apply_interaction_generator(
        fit_subset: pd.DataFrame,
        transform_subsets: list,
        features: np.ndarray,
        degree: int = 2
) -> list:
    """ Features Interaction Generator

    The main function aims to extend the datasets,
    present in subsets argument, with interaction features,
    until the degree argument is reached.

    Args:
        fit_subset: Data, in which the object will be fitted
        transform_subsets: Data, to be extended with interactions
        features: Features to consider for the interactions
        degree: Maximum, not included, degree of the interactions

    Returns:
        Extended datasets
    """

    polynomial_object = PolynomialFeatures(
        degree=degree,
        interaction_only=True,
        include_bias=False
    ).fit(fit_subset[features])

    outputs = list()

    transform_subsets.insert(0, fit_subset)

    for this_set in transform_subsets:

        transformed_array = polynomial_object.transform(this_set[features])

        transformed_set = pd.DataFrame(
            data=transformed_array,
            columns=polynomial_object.get_feature_names(features)
        )

        this_output = pd.merge(
            this_set,
            transformed_set,
            how='inner'
        )

        this_output = this_output.loc[:, ~this_output.columns.duplicated()]

        outputs.append(this_output)

    return outputs


def apply_scaler(
        fit_subset: pd.DataFrame,
        transform_subset: list,
        features: np.ndarray,
        scaler=StandardScaler()
) -> list:
    """ Feature Scaling

    The main function aims to scale the features, given as argument.
    The scaler will be fitted by means of the fit_subset, and transform
    not only this subset but the others present in transform_subset list.
    
    Args:
        scaler: Sklearn scaler instance
        fit_subset: Data, in which the scaler will be fitted
        transform_subset: Data, to be scaled/transformed besides the fit_subset
        features: Features to be considered for the scaler

    Returns:
        Transformed data, in which the order is fit_subset and transform_subset
    """

    outputs = list()

    scaler.fit(fit_subset[features])

    transform_subset.insert(0, fit_subset)

    for this_set in transform_subset:

        this_set_cp = this_set.copy()

        transformed_array = scaler.transform(this_set_cp[features])

        this_set_cp[features] = transformed_array

        outputs.append(this_set_cp)

    return outputs


def apply_pca(
        fit_subset: pd.DataFrame,
        transform_subset: list,
        features: np.ndarray,
        min_variance_explained: float = 0.9,
) -> list:
    """Principal Components Analysis application

    The function aims to fit the PCA object in the fit_subset,
    and then transform each of the datasets in the transform_subset.


    Args:
        fit_subset: Data where PCA will be fitted and transformed
        transform_subset: Dataset(s) to be transformed by the fitted PCA object
        features: Continuous features, based on which PCA
        will be fit or transformed.
        min_variance_explained: Minimum ratio of variance
        of the dataset that the kept components should explain

    Returns:
        A list containing the transformed datasets with or without the
        fitted PCA object.
    """
    flag = True

    pca_instance = PCA(n_components=1).fit(fit_subset[features])

    while flag:

        if sum(pca_instance.explained_variance_ratio_) < min_variance_explained:

            components_number = pca_instance.n_components

            pca_instance = PCA(n_components=components_number+1).fit(fit_subset[features])

        else:

            flag = False

    output = list()

    transform_subset.insert(0, fit_subset)

    for this_dataset in transform_subset:

        this_scores = pca_instance.transform(this_dataset[features])

        this_df = pd.DataFrame(
            data=this_scores,
            columns=[f'Y_{i+1}' for i in range(this_scores.shape[1])]
        )

        this_df[
            np.setdiff1d(this_dataset.columns, features)
        ] = this_dataset[
            np.setdiff1d(this_dataset.columns, features)
        ].copy()

        output.append(this_df)

    return output
