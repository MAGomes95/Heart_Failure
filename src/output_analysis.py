from joblib import load

import numpy as np
import pandas as pd

from src.visualization.error_analysis import plot_range_comparison

if __name__ == '__main__':

    # Data and Model Loading

    test = pd.read_csv(
        filepath_or_buffer='../data/silver/tree_based/test.csv',
        index_col=0
    )

    xgboost_model = load('../models/trained/cb.joblib')

    # Predictions Generation and Error Analysis Performing

    test['predictions'] = xgboost_model.predict(
        test[np.setdiff1d(test.columns, 'target')]
    )
    test['predictions'] = test['target'] == test['predictions']

    original_features = [
        feature for feature in test.columns if len(feature.split(' ')) == 1
    ]

    plot_range_comparison(
        data=test,
        features=np.setdiff1d(original_features, ['target', 'predictions']),
        classification_feature='predictions',
        saving_path='../reports/figures/cb/error_analysis/test'
    )


