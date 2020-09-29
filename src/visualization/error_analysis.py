import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_range_comparison(
        data: pd.DataFrame,
        features: np.ndarray,
        classification_feature: str,
        saving_path: str
) -> None:
    """Compare value range among instances

    The main function aims to, for each feature
    present in features, compare the rang of values
    of the misclassified instances the correctly
    classified ones by means of a kde visualization

    Args:
        data: Data, containing the features to compare
        features: Features to consider for the comparison
        classification_feature: Prediction output from the model
        saving_path: Path to the folder, where the visualizations
        will be stores

    Note that the data must have a classification_feature that identifies
    correctly classified instances from the misclassified:
    0 - Misclassified // 1 - Correctly classified

    Returns:
        The function generates a visualization for each feature
        and saves it in a saving_path
    """

    for feature in features:

        # Continuous Features

        if data[feature].nunique() > 2:

            plot = sns.displot(
                data=data,
                x=feature,
                hue=classification_feature,
                kind='kde'
            )

        # Categorical Features

        else:

            to_plot = pd.DataFrame(
                data=np.transpose(
                    [[0, 0, 1, 1], [0, 1, 0, 1], [

                        data.loc[(data[feature] == 0) & (data[classification_feature] == 0)].shape[0]
                        / data.loc[data[feature] == 0].shape[0],

                        data.loc[(data[feature] == 0) & (data[classification_feature] == 1)].shape[0]
                        / data.loc[data[feature] == 0].shape[0],

                        data.loc[(data[feature] == 1) & (data[classification_feature] == 0)].shape[0]
                        / data.loc[data[feature] == 1].shape[0],

                        data.loc[(data[feature] == 1) & (data[classification_feature] == 1)].shape[0]
                        / data.loc[data[feature] == 1].shape[0]
                    ]]
                ),
                columns=[feature, 'classification', 'proportion']
            )

            plot = sns.barplot(
                x=feature,
                y='proportion',
                hue='classification',
                data=to_plot
            )

        try:
            plt.savefig(
                f'{saving_path}/{feature}.png'
            )
        except FileNotFoundError:

            feature_name_parts = feature.split('/')
            new_feature_name = ''.join(feature_name_parts)

            plt.savefig(
                f'{saving_path}/{new_feature_name}.png'
            )

        plt.close('all')
