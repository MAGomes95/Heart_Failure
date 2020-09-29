import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def membership_feature(
        fit_subset: pd.DataFrame,
        transform_subsets: list,
        features: np.ndarray,
        clustering_model=KMeans(),
        n_clusters: int = 3
) -> list:
    """Expanding the dataset with a clustering feature

    The function aims to extend the given dataset, with
    a additional feature that consists in the membership
    attributed to each instance by a clustering model with
    parameter number of clusters to be n_clusters

    Args:
        fit_subset: data, based on which, the model will be fitted
        transform_subsets: datasets to be extended with the membership feature
        features: features to consider for clustering the instances
        clustering_model: Clustering model. By default, KMeans
        n_clusters: Number of clusters. By default, 3

    Returns:
        Extend datasets with the additional feature
    """

    clustering_model.n_clusters = n_clusters

    fitted_model = clustering_model.fit(fit_subset[features])

    transform_subsets.insert(0, fit_subset)

    outputs = []

    for data in transform_subsets:

        data_cp = data.copy()

        data_cp['memberships'] = fitted_model.predict(data[features])

        outputs.append(data_cp)

    return outputs
