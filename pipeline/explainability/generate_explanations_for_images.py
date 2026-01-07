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
import shutil
import re
import os
from utils import __get_privlex_project_path__, __get_results_path__
from dataset import get_dataset_train_test_val_select
from pipeline.train_linear_layer_on_concept_scores.simple_classifiers_utils import eval_simple_classifier, normalise_input_data
from pipeline.concept_scores_extraction import extract_concept_scores
from utils import load_json
from pipeline.train_linear_layer_on_concept_scores.train_linear_layer_on_concept_scores \
    import load_linear_layer_model_by_experiment_name, load_and_parse_train_config
from utils import find_img_path_by_photoid


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'
RESULTS_PATH = __get_results_path__()
TRAIN_CONFIGS_PATH = PRIVLEX_PROJECT_PATH + 'configs/'

def split_camel_case(word):
    return re.sub(r'(?<=[a-zA-Z])(?=[A-Z][a-z])', ' ', word)  # Split before last uppercase letter if followed by lowercase


def save_all_dpv_related_info_for_the_image(img_df, norm_param, model, results_dir, set_type, n_top_highest=20, n_each_contrib=8):
    '''
    :param results_dir: directory of the model
    :param img_df: img_df = df_train.iloc[i]
    :param norm_param: normalisation parameters for img_df
    :param set_type:
    :return:
    '''

    img_val = (img_df.to_numpy()[2:] - norm_param[0]) / norm_param[1]
    img_val = pd.DataFrame(np.expand_dims(img_val, 0), columns=img_df.index[2:])
    pred, priv_prob = model.predict(img_val)[0], model.predict_proba(img_val)[0, 1]

    phid = int(img_df['photoid']) if type(img_df['photoid']) == np.float64 else img_df['photoid']
    privacy = int(img_df['privacy'])

    try:
        img_path = find_img_path_by_photoid(phid)
    except:
        img_path = 'Path not found'
        print(f'File {img_path} does not exist')

    image_dict = {'photoid': str(phid),
                  'privacy':privacy,
                  'model_pred': int(pred),
                  'model_priv-prob': priv_prob,
                  'dataset': set_type}

    dpv_names = img_df.index[2:]
    dpv_sim_values = img_df.to_numpy()[2:]
    sorted_idx = np.argsort(dpv_sim_values)[::-1]
    image_dict[f'dpv_{n_top_highest}_highest_cos-sim'] = tuple(dpv_names[sorted_idx[:n_top_highest]])

    model_weights = model.coef_[0]

    pos_concepts = {}
    neg_concepts = {}
    zero_concepts = {}

    concepts_contrib = img_val * model_weights

    for idx in sorted_idx:
        concept_model_weights = np.round(model_weights[idx], 4)
        name = dpv_names[idx]
        similarity = np.round(dpv_sim_values[idx], 4)
        contribution = float(np.round(concepts_contrib.to_numpy()[0][idx], 4))
        concept_dict = {'similarity': similarity, 'model_weight': concept_model_weights,
                                  'contribution': contribution}
        if concept_model_weights > 0 and len(pos_concepts) < n_each_contrib:
            pos_concepts[name] = concept_dict
        elif concept_model_weights < 0 and len(neg_concepts) < n_each_contrib:
            neg_concepts[name] = concept_dict
        elif concept_model_weights == 0 and len(zero_concepts) < n_each_contrib:
            zero_concepts[name] = concept_dict

        if len(pos_concepts) >= n_each_contrib and len(neg_concepts) >= n_each_contrib\
                and len(zero_concepts) >= n_each_contrib:
            break


    image_dict['pos_contributing_concepts'] = pos_concepts
    image_dict['neg_contributing_concepts'] = neg_concepts
    image_dict['zero_contributing_concepts'] = zero_concepts

    if pred == privacy:
        curr_results_dir = results_dir + 'corr-cls/'
    else:
        curr_results_dir = results_dir + 'miss-cls/'

    highest_concept = dpv_names[sorted_idx[0]]
    curr_results_dir = curr_results_dir + highest_concept + '/'

    if not os.path.exists(curr_results_dir):
        os.makedirs(curr_results_dir)

    curr_results_filename = curr_results_dir + f'{phid}_priv-{privacy}_ds-{set_type}'

    import json
    with open(curr_results_filename + '.json', 'w') as f:
        json.dump(image_dict, f)

    if os.path.exists(img_path):
        shutil.copy(img_path, curr_results_filename + '.' + img_path.split('.')[-1])
    print(curr_results_filename.split('/')[-1])


def save_detected_concepts_for_images_for_privlex(image_type_to_display, n_images_to_display,
                                             PRETRAINED_MODEL, dataset_name, concept_type,
                                             config_name='train_bipd_v2_logreg-l1score-Cbelow1-norm[0, 1]',
                                             n_concepts_to_save=30):

    concept_scores_df = extract_concept_scores(PRETRAINED_MODEL=PRETRAINED_MODEL,
                                               dataset_name=dataset_name,
                                               concept_type=concept_type)
    df_train, df_test, df_val = get_dataset_train_test_val_select(dataset_name, concept_scores_df, include_val=True)

    config_path = TRAIN_CONFIGS_PATH + config_name + '.json'

    train_name, \
    n_trials, metric_to_optimize, normalize, set_to_optimize_on, \
    parameters_to_optimize, other_simple_classifier_parameters = load_and_parse_train_config(config_path)

    model = load_linear_layer_model_by_experiment_name(PRETRAINED_MODEL, dataset_name, concept_type,
                                                       config_name=config_name)

    experiment_tag = f"{dataset_name} linear_layer_{train_name}__{concept_type}__with__{PRETRAINED_MODEL.split('/')[-1]}"
    results_dir = RESULTS_PATH + f'{dataset_name}/{experiment_tag}/'


    (train_metrics_dict, test_metrics_dict, val_metrics_dict),\
            (y_pred_train, y_pred_test, y_pred_val) = eval_simple_classifier(experiment_tag, df_train, df_test, df_val,
                                                                             model, print_results=True, normalize=normalize,
                                                                             return_predictions=True)


    if image_type_to_display == 'TP':
        condition = (df_test.privacy == y_pred_test) & (df_test.privacy == 1)
    elif image_type_to_display == 'TN':
        condition = (df_test.privacy == y_pred_test) & (df_test.privacy == 0)
    elif image_type_to_display == 'FP':
        condition = (df_test.privacy != y_pred_test) & (df_test.privacy == 0)
    elif image_type_to_display == 'FN':
        condition = (df_test.privacy != y_pred_test) & (df_test.privacy == 1)
    elif type(image_type_to_display) == list or type(image_type_to_display) == np.ndarray:
        condition = df_test.photoid.isin(image_type_to_display)
        condition_val = df_val.photoid.isin(image_type_to_display)
        condition_train = df_train.photoid.isin(image_type_to_display)
    else:
        raise Exception(f'Unknown image type: {type(image_type_to_display)}')

    df_test_selected = df_test[condition]
    if not (type(image_type_to_display) == list or type(image_type_to_display) == np.ndarray):
        df_test_selected = df_test_selected.loc[
            np.random.choice(df_test_selected.index, np.min([n_images_to_display, len(df_test_selected)]),
                             replace=False)]

    df_train_norm, df_test_norm, df_val_norm, normalisation_params = normalise_input_data(df_train, df_test, df_val,
                                                                             normalize=normalize,
                                                                             return_normalisation_params=True,
                                                                             return_dataframes=True)


    for idx in range(len(df_test_selected)):
        print(f'saving json for {df_test_selected.iloc[idx][0]} image')
        img_df = df_test_selected.iloc[idx]
        save_all_dpv_related_info_for_the_image(img_df, normalisation_params,
                                                model, results_dir, set_type='test',
                                                n_top_highest=n_concepts_to_save,
                                                n_each_contrib=n_concepts_to_save)

    if type(image_type_to_display) == list or type(image_type_to_display) == np.ndarray:
        df_train_selected = df_train[condition_train]
        df_val_selected = df_test[condition_val]
        for set_type, df_selected in [['train', df_train_selected], ['val', df_val_selected]]:
            for idx in range(len(df_selected)):
                print(f'saving json for {df_selected.iloc[idx][0]} image')
                img_df = df_selected.iloc[idx]
                save_all_dpv_related_info_for_the_image(img_df, normalisation_params,
                                                        model, results_dir, set_type=set_type,
                                                        n_top_highest=n_concepts_to_save,
                                                        n_each_contrib=n_concepts_to_save)




def print_concepts_corresponding_to_an_image__with_colors_from_related_info_dict(
        image_dict, n_top_highest, n_words_to_print=None, threshold_value=0.235, print_dict_for_each_concepts=True):

    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"  # Reset color to default

    # Print words with corresponding colors
    flag = 0
    for word in image_dict[f'dpv_{n_top_highest}_highest_cos-sim'][:n_words_to_print]:
        formatted_word = split_camel_case(word)

        if word in image_dict['pos_contributing_concepts']:
            word_dict = image_dict['pos_contributing_concepts'][word]
            word_color = RED
        elif word in image_dict['neg_contributing_concepts']:
            word_dict = image_dict['neg_contributing_concepts'][word]
            word_color = GREEN
        elif word in image_dict['zero_contributing_concepts']:
            word_dict = image_dict['zero_contributing_concepts'][word]
            word_color = ''
        else:
            raise Exception(f'Concept {word} not found in positive, negative or zero contributing')

        if word_dict['similarity'] <= threshold_value and flag == 0:
            print('-----------------------------------------------------------------------\n')
            flag = 1
        if print_dict_for_each_concepts:
            print(f"{word_color}{formatted_word}{RESET}: {word_dict}")
        else:
            print(f"{word_color}{formatted_word}{RESET}")



def find_json_files(base_dir, starts_with):
    matching_files = []

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.startswith(str(starts_with)) and fname.endswith(".json"):
                full_path = os.path.join(root, fname)
                matching_files.append(full_path)

    return matching_files



def __get_parser__():
    parser = argparse.ArgumentParser(description="Extract image embeddings using PRETRAINED_MODEL.")
    parser.add_argument("--PRETRAINED_MODEL", type=str, required=True,
                        help="Pretrained model path or name (e.g. openai/clip-vit-base-patch32, google/siglip2-base-patch32-256, facebook/flava-full)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g. privacyalert, vispr)")
    parser.add_argument("--concept_type", type=str, required=True, help="Name of the concept type "
                                                                        "(e.g. dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot)")
    parser.add_argument("--classifier_config_name", type=str, required=True,
                        help="Name of the train config (e.g. train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_privacyalert)")
    def process_list_of_photoids(list_of_photoids):
        processed_list_of_photoids = []

        for item in list_of_photoids.split(","):
            photoid = item.strip()
            try: photoid = int(photoid)
            except: photoid = photoid
            processed_list_of_photoids.append(photoid)
        return processed_list_of_photoids

    parser.add_argument("--list_of_photoids", type=process_list_of_photoids, required=False, default=None,
                        help="Comma separated list of photoids for which the identified concepts will be printed out (e.g. '49910698552, 50783798341')")

    return parser



if __name__ == '__main__':
    args = __get_parser__().parse_args()

    dataset_name = args.dataset_name
    concept_type = args.concept_type
    PRETRAINED_MODEL = args.PRETRAINED_MODEL
    classifier_config_name = args.classifier_config_name
    list_of_photoids = args.list_of_photoids

    if list_of_photoids is None:
        # if no list of photoids is provided, the explanations are going to be saved for 50 random images for true/false positive/negative,
        for type_of_images in ['TP', 'FP', 'TN', 'FN']:
            save_detected_concepts_for_images_for_privlex(
                image_type_to_display='TP', n_images_to_display=50,
                PRETRAINED_MODEL=PRETRAINED_MODEL, dataset_name=dataset_name,
                concept_type=concept_type,
                n_concepts_to_save=30,
                config_name=classifier_config_name)

    else:
        save_detected_concepts_for_images_for_privlex(
            image_type_to_display=list_of_photoids, n_images_to_display=50,
            PRETRAINED_MODEL=PRETRAINED_MODEL, dataset_name=dataset_name,
            concept_type=concept_type,
            n_concepts_to_save=30,
            config_name=classifier_config_name)


        base_dir = RESULTS_PATH + f"{dataset_name}/{dataset_name} linear_layer_{classifier_config_name}__{concept_type}__with__{PRETRAINED_MODEL.split('/')[-1]}"
        for img_photoid in list_of_photoids:
            image_path = find_json_files(base_dir, img_photoid)
            if len(image_path) == 0:
                raise Exception(f"No images {img_photoid} found in {base_dir}")
            elif len(image_path) > 1:
                raise Exception(f"Multiple images {img_photoid} found in {base_dir}")
            else:
                image_path = image_path[0]
            image_dict = load_json(image_path)
            print_dict_for_each_concepts = True
            print(f'\n\nConcepts detected for the image {img_photoid}:')
            print_concepts_corresponding_to_an_image__with_colors_from_related_info_dict(image_dict, n_top_highest=30,
                                                                                         n_words_to_print=3,
                                                                                         print_dict_for_each_concepts=print_dict_for_each_concepts,
                                                                                         threshold_value=0.245)
