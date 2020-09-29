import shap
import pandas as pd
import matplotlib.pyplot as plt


def instance_explanation(
        model,
        data: pd.DataFrame,
        instance_id: int,
        saving_path: str
) -> None:
    """Plot predictive explanations for a single instance

    The functions aims to create and save a matplotlib visualization
    explanation for the instance of the data, whose location as been
    given as argument.

    Args:
        model: Scikit-learn, XGBoost, CatBoost or LighGBM model
        data: data based on which prediction will be generated
        instance_id: location/index of the instance
        saving_path: Path and filename for explanation saving

    Returns:
        The visualization will be saved in the reports/figures folder
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    figure = shap.force_plot(
        explainer.expected_value,
        shap_values[instance_id, :],
        data.iloc[instance_id, :],
        show=False,
        matplotlib=True
    )

    plt.savefig(
        saving_path
    )


def summary_explanation(
        model,
        saving_path: str,
        data: pd.DataFrame
) -> None:
    """Plot feature explanations

    The main function attempts to create and save the feature
    explanations based on the model and data, given as argument,
    with emphasis on the feature values.

    Args:
        model: Scikit-learn, XGBoost or CatBoost model
        saving_path: Path and filename for summary saving
        data: Data, based on which explanations will be computed

    Returns:
        The visualization will be saved in reports/figure folder
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    figure = shap.summary_plot(
        shap_values, data, plot_type='bar', feature_names=data.columns, show=False
    )

    plt.savefig(
        saving_path
    )


def dependency_explanation(
        model,
        data: pd.DataFrame,
        main_feature: str,
        saving_path: str,
        interaction_feature: str = None
) -> None:
    """Plot feature dependency explanations

    The functions aims to create and save a dependency plot
    explanation, in which singular explanations are plotted
    as a function of those two features given

    Args:
        model: Scikit-learn, XGBoost or CatBoost models
        data: Data, upon which, predictions and expandability will be generated
        main_feature: Principal feature, upon which, explanabilities will be generated
        saving_path: Path and filename for explanation saving
        interaction_feature: Secondary feature, if any, whose importance will be plot in top of the main_feature

    Returns:
        Visualization plot will be saved in reports/figures folder
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    if interaction_feature is None:

        figure = shap.dependence_plot(
            main_feature, shap_values, data, show=False
        )

    else:

        figure = shap.dependence_plot(
            main_feature, shap_values, data, interaction_index=interaction_feature, show=False
        )

    plt.savefig(
        saving_path
    )
