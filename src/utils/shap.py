import warnings
import logging

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import shap


meta_system_replacements = {
    'meta_system_0': 'Oesophagus',
    'meta_system_1': 'Stomach',
    'meta_system_2': 'Colorectum',
    'meta_system_3': 'Liver',
    'meta_system_4': 'Pancreas'
}


def plot_shap_values(model, X_test, X_train, feature_names, explainer_class='deep'):
    #X_test_df = pd.DataFrame(X_test, columns=feature_names)
    #X_train_df = pd.DataFrame(X_train, columns=feature_names)
    #X_test_df['meta_system'].replace(meta_system_replacements)
    #X_train_df['meta_system'].replace(meta_system_replacements)
    def get_shap_values(test_data, train_data):
        if explainer_class == 'deep':
            explainer = shap.DeepExplainer(model, train_data)
        else:
            explainer = shap.KernelExplainer(model, train_data[:100])
        #return explainer(test_data)
        return explainer.shap_values(test_data)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Test SHAP values
        test_shap_values = get_shap_values(X_test, X_train)
        #test_shap_values = get_shap_values(X_test_df, X_train_df)
        logging.debug('SHAP Values')
        logging.debug(test_shap_values)
        fig = plt.figure()
        shap.summary_plot(test_shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
        #shap.plots.bar(test_shap_values, show=False)
        plt.tight_layout()
        wandb.log({'SHAP Value Bars': wandb.Image(fig)})
        plt.close()

        fig = plt.figure()
        shap.summary_plot(test_shap_values[0], X_test, feature_names=feature_names, plot_type='dot', show=False)
        #shap.plots.bar(test_shap_values, show=False)
        plt.tight_layout()
        wandb.log({'SHAP Value Dots': wandb.Image(fig)})
        plt.close()

        # fig = plt.figure()
        # shap.plots.bar(test_shap_values.cohorts(X_test_df['meta_system'].abs.mean(axis=0)), show=False)
        # plt.tight_layout()
        # wandb.log({'SHAP Value Bars (By Organ)': wandb.Image(fig)})
        # plt.close()

        # fig = plt.figure()
        # shap.plots.beeswarm(test_shap_values, show=False)
        # plt.tight_layout()
        # wandb.log({'SHAP Value Beeswarm': wandb.Image(fig)})
        # plt.close()

        shap_value_df = pd.DataFrame(columns=feature_names)
        for i in range(len(test_shap_values)):
            single_output_shap_values = pd.Series(np.abs(test_shap_values[i]).mean(0), name=str(i), index=feature_names)
            shap_value_df = pd.concat([shap_value_df, single_output_shap_values.to_frame().T])

        wandb.log({'SHAP Values': wandb.Table(dataframe=shap_value_df)})
