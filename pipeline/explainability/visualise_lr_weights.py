#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import __get_privlex_project_path__, __get_results_path__
from pipeline.train_linear_layer_on_concept_scores.train_linear_layer_on_concept_scores \
    import load_linear_layer_model_by_experiment_name, load_and_parse_train_config


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'
RESULTS_PATH = __get_results_path__()
TRAIN_CONFIGS_PATH = PRIVLEX_PROJECT_PATH + 'configs/'


paper_model_colors = [np.array(c) / 255 for c in [(232, 166, 108), (49, 90, 135), (135, 79, 117)]]
paper_vispr_description_colors = [np.array(c) / 255 for c in [(96, 187, 183), (255, 187, 183),  (0, 255, 183), ]]



def create_top_indexes_from_str(models_weights_df, top_indexes_type, top_coefs_number=20):
    if type(top_indexes_type) == np.array or type(top_indexes_type) == list or type(top_indexes_type) == np.ndarray:
        return top_indexes_type

    models_weights_list = models_weights_df.iloc[:, 2:]

    models_weights_list = models_weights_list.to_numpy().T

    if top_indexes_type is None:
        top_indexes = np.arange(models_weights_list.shape[1])
    elif top_indexes_type == 'highest_average_private_class_weight':
        mean = np.mean(models_weights_list, axis=0)
        top_indexes_all = np.argsort(mean)[::-1]
        top_indexes = []
        for t in top_indexes_all:
            if models_weights_list[0][t] >= 0 and models_weights_list[1][t] >= 0 and models_weights_list[-1][t] >= 0:
                top_indexes.append(t)

            if len(top_indexes) >= top_coefs_number:
                break
    elif top_indexes_type == 'highest_average_public_class_weight':
        mean = np.mean(models_weights_list, axis=0)
        top_indexes_all = np.argsort(mean)
        top_indexes = []
        for t in top_indexes_all:
            if models_weights_list[0][t] <= 0 and models_weights_list[1][t] <= 0 and models_weights_list[-1][t] <= 0:
                top_indexes.append(t)
            if len(top_indexes) >= top_coefs_number:
                break
    elif top_indexes_type == 'highest_std_for_models_weights':
        std = np.std(models_weights_list, axis=0)
        top_indexes = np.argsort(std)[::-1][:top_coefs_number]
    top_indexes = np.array(top_indexes)
    return top_indexes


def plot_graph_with_points_from_df_of_concepts(concepts_and_corresponding_vals_df, top_indexes_type,
                                               name_prefix,
                                               results_path, top_coefs_number=None, figsize=(6, 10),
                                               colors=paper_model_colors, xlabel="Weight", x_lim=None):


    fig, ax = plt.subplots(figsize=figsize)

    if top_coefs_number is None:
        top_coefs_number = len(concepts_and_corresponding_vals_df)

    top_indexes = create_top_indexes_from_str(concepts_and_corresponding_vals_df, top_indexes_type, top_coefs_number=top_coefs_number)

    model_names = concepts_and_corresponding_vals_df.columns[2:].to_numpy()

    for i in range(len(top_indexes)):
        top_idx = top_indexes[i]
        row = concepts_and_corresponding_vals_df.iloc[top_idx]
        values = row[2:].to_numpy()
        y = len(top_indexes) - 1 - i # same y for each concept
        print(row[0], y)

        # Plot line connecting values
        ax.plot([np.min(values), np.max(values)], [y, y], color="black", lw=1, zorder=1)

        # Plot scatter points
        for model_idx in range(len(model_names))[::-1]:
            ax.scatter(values[model_idx], y, color=colors[model_idx], s=80, label=model_names[model_idx] if i == 0 else "", zorder=2)


    concepts_shown = concepts_and_corresponding_vals_df.iloc[top_indexes][::-1]['Term']
    plt.yticks(np.arange(top_coefs_number), concepts_shown,
               fontsize=10)
    ax.set_xlabel(xlabel)
    ax.legend()
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    if type(top_indexes_type) is str:
        ax.set_title(f"{name_prefix} {top_indexes_type}")
    else:
        ax.set_title(f"{name_prefix}")
    plt.tight_layout()
    results_file_path = results_path + name_prefix
    if type(top_indexes_type) is str:
        results_file_path = results_file_path + f'_{top_indexes_type}'
    plt.savefig(results_file_path + f'.png')
    # plt.savefig(results_file_path + f'.svg')
    plt.show()

    for i in top_indexes:
        print(concepts_and_corresponding_vals_df.iloc[i, 1])


def make_dataframe_for_concepts_and_corresponding_model_weights(dataset_names,
                                                                config_names,
                                                                concepts_df,
                                                                concept_type,
                                                                PRETRAINED_MODEL,
                                                                normalise_coefs,
                                                                models_names_to_represent=None):
    if models_names_to_represent is None:
        models_names_to_represent = dataset_names

    models_weights_df = concepts_df[['Term', 'text']]
    for i in range(len(dataset_names)):
        curr_dataset_model = load_linear_layer_model_by_experiment_name(
            PRETRAINED_MODEL=PRETRAINED_MODEL, dataset_name=dataset_names[i], concept_type=concept_type, config_name=config_names[i])
        if normalise_coefs:
            curr_dataset_model_coef = curr_dataset_model.coef_[0] / np.abs(curr_dataset_model.coef_[0]).max()
        else:
            curr_dataset_model_coef = curr_dataset_model.coef_[0]

        models_weights_df[models_names_to_represent[i]] = curr_dataset_model_coef

    return models_weights_df



def __get_parser__():
    parser = argparse.ArgumentParser(description="Extract image embeddings using PRETRAINED_MODEL.")
    parser.add_argument("--PRETRAINED_MODEL", type=str, required=True,
                        help="Pretrained model path or name (e.g. openai/clip-vit-base-patch32, google/siglip2-base-patch32-256, facebook/flava-full)")
    parser.add_argument("--dataset_names", type=lambda s: [item.strip() for item in s.split(", ")], required=True,
                        help="List of dataset names for which the weights should be visualised (e.g. privacyalert, vispr)")
    parser.add_argument("--concept_type", type=str, required=True, help="Name of the concept type "
                                                                        "(e.g. dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot)")
    parser.add_argument("--classifier_config_names", type=lambda s: [item.strip() for item in s.split("; ")],
                        required=True, help="List of train config names for the datasets separated by ;")

    return parser



if __name__ == '__main__':
    args = __get_parser__().parse_args()

    dataset_names = args.dataset_names
    concept_type = args.concept_type
    PRETRAINED_MODEL = args.PRETRAINED_MODEL
    classifier_config_names = args.classifier_config_names

    concepts_df = pd.read_csv(data_path + f'concept_types/{concept_type}.csv')

    models_weights_df = make_dataframe_for_concepts_and_corresponding_model_weights(dataset_names,
                                                                                    config_names=classifier_config_names,
                                                                                    concept_type=concept_type,
                                                                                    PRETRAINED_MODEL=PRETRAINED_MODEL.split('/')[-1],
                                                                                    normalise_coefs=True,
                                                                                    concepts_df=concepts_df,
                                                                                    models_names_to_represent=dataset_names)

    print('Visualising model wights for concepts that have the highest weight for private and public class on average.')
    print('Visualising model wights for concepts that have the highest weight STD across the datasets.')
    top_indexes_types = ['highest_average_private_class_weight', 'highest_average_public_class_weight',
                         'highest_std_for_models_weights']
    for top_indexes_type in top_indexes_types:
        plot_graph_with_points_from_df_of_concepts(models_weights_df,
                                                   top_indexes_type=top_indexes_type,
                                                   name_prefix=f'datasets-{dataset_names}__lr_weights__{concept_type}',
                                                   results_path=RESULTS_PATH,
                                                   top_coefs_number=20, figsize=(6, 8), xlabel="Weight")

    top_indexes_type = np.argsort(models_weights_df['Term'].to_numpy())
    plot_graph_with_points_from_df_of_concepts(models_weights_df,
                                               top_indexes_type=top_indexes_type,
                                               name_prefix=f'datasets-{dataset_names}__all_lr_weights__{concept_type}',
                                               results_path=RESULTS_PATH,
                                               top_coefs_number=None, figsize=(6, 25), xlabel="Weight")

