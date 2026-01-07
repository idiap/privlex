#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import os.path
import argparse
import json
import pickle
from utils import __get_privlex_project_path__, __get_results_path__
from utils import load_json
from dataset import get_dataset_train_test_val_select
from pipeline.train_linear_layer_on_concept_scores.simple_classifiers_utils import optimize_hyperparameters_with_optuna, \
    run_simple_classifier, eval_simple_classifier, run_classifier_training_across_seeds
from pipeline.concept_scores_extraction import extract_concept_scores


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'
TRAIN_CONFIGS_PATH = PRIVLEX_PROJECT_PATH + 'configs/'
RESULTS_PATH = __get_results_path__()


def optimise_and_train_bipd_for_a_dataset(tag, df_train, df_test, df_val,
                                          metric_to_optimize, normalize,
                                          n_trials, set_to_optimize_on,
                                          parameters_to_optimize,
                                          other_simple_classifier_parameters):
    dataset_name = tag.split(' ')[0]
    random_state = 15
    normalize_during_train = normalize


    if normalize == 'min_max_to_0_1_across_datasets':
        raise NotImplementedError('Normalisation for min_max_to_0_1_across_datasets not implemented yet')

    best_hyperparams, average_df = optimize_hyperparameters_with_optuna(set_to_optimize_on, metric_to_optimize,
                                                                        parameters_to_optimize,
                                                                        other_simple_classifier_parameters,
                                                                        tag,
                                                                        df_train, df_test, df_val,
                                                                        random_state, normalize=normalize_during_train,
                                                                        n_trials=n_trials, timeout=500)


    simple_classifier_parameters = {}
    for k in other_simple_classifier_parameters:
        simple_classifier_parameters[k] = other_simple_classifier_parameters[k]
    for k in best_hyperparams.params:
        simple_classifier_parameters[k] = best_hyperparams.params[k]


    (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val), model \
         = run_simple_classifier(tag=tag,
                                             df_train=df_train, df_test=df_test, df_val=df_val,
                                             simple_classifier_parameters=simple_classifier_parameters,
                                             random_state=random_state,
                                             normalize=normalize_during_train,
                                             print_results=True,
                                             return_predictions=True,
                                             return_model=True, fit_parameters={})

    print('number of zero params: ', (model.coef_ == 0).sum())

    results_dir = RESULTS_PATH + f'{dataset_name}/{tag}/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filename = results_dir + 'model' + f'_{tag}.sav'
    pickle.dump(model, open(filename, 'wb'))
    return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val), model


def load_and_parse_train_config(config_path):
    train_config = load_json(config_path)
    train_name = train_config["name"]
    print(f'Training config {train_name} created on {train_config["date_created"]}')
    print(json.dumps(train_config, indent=4))

    n_trials = train_config["n_trials"]
    metric_to_optimize = train_config["metric_to_optimize"]
    normalize = train_config["normalize"]
    set_to_optimize_on = train_config["set_to_optimize_on"]

    parameters_to_optimize = train_config["parameters_to_optimize"]
    other_simple_classifier_parameters = train_config["other_simple_classifier_parameters"]
    return train_name, \
           n_trials, metric_to_optimize, normalize, set_to_optimize_on,\
           parameters_to_optimize, other_simple_classifier_parameters


def load_linear_layer_model_by_experiment_name(PRETRAINED_MODEL, dataset_name, concept_type,
                                                config_name):
    config_path = TRAIN_CONFIGS_PATH + config_name + '.json'
    train_name, \
        n_trials, metric_to_optimize, normalize, set_to_optimize_on, \
        parameters_to_optimize, other_simple_classifier_parameters = load_and_parse_train_config(config_path)

    experiment_tag = f"{dataset_name} linear_layer_{train_name}__{concept_type}__with__{PRETRAINED_MODEL.split('/')[-1]}"

    results_dir = RESULTS_PATH + f'{dataset_name}/{experiment_tag}/model' + f'_{experiment_tag}.sav'

    with open(results_dir, 'rb') as f:
        model = pickle.load(f)
    return model


def run_model(PRETRAINED_MODEL, dataset_name, concept_type, config_name):
    concept_scores_df = extract_concept_scores(PRETRAINED_MODEL=PRETRAINED_MODEL,
                                               dataset_name=dataset_name,
                                               concept_type=concept_type)
    df_train, df_test, df_val = get_dataset_train_test_val_select(dataset_name, concept_scores_df, include_val=True)

    config_path = TRAIN_CONFIGS_PATH + config_name + '.json'

    train_name, \
    n_trials, metric_to_optimize, normalize, set_to_optimize_on, \
    parameters_to_optimize, other_simple_classifier_parameters = load_and_parse_train_config(config_path)

    experiment_tag = f"{dataset_name} linear_layer_{train_name}__{concept_type}__with__{PRETRAINED_MODEL.split('/')[-1]}"

    (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val), model = optimise_and_train_bipd_for_a_dataset(
        experiment_tag, df_train, df_test, df_val, metric_to_optimize, normalize,
                                          n_trials, set_to_optimize_on,
                                          parameters_to_optimize,
                                          other_simple_classifier_parameters)


def evaluate_model(PRETRAINED_MODEL, dataset_name, concept_type,
              config_name='train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]',
                   return_predictions=True, return_model=True):
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

    (train_metrics_dict, test_metrics_dict, val_metrics_dict),\
            (y_pred_train, y_pred_test, y_pred_val) = eval_simple_classifier(experiment_tag, df_train, df_test, df_val,
                                                                             model, print_results=True, normalize=normalize,
                                                                             return_predictions=True)
    if return_predictions:
        if return_model:
            return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val), model
        return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val)


def evauate_model_with_multiple_seeds(PRETRAINED_MODEL, dataset_name, concept_type,
              config_name='train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]',
              random_seeds=[1347, 20, 4678, 829, 1558, 39, 991, 96, 4, 435]):
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

    simple_classifier_parameters = {}
    for k in other_simple_classifier_parameters:
        simple_classifier_parameters[k] = other_simple_classifier_parameters[k]
    for k in parameters_to_optimize:
        simple_classifier_parameters[k] = getattr(model,k)

    from dataset import TagsDFWrapper

    average_df = run_classifier_training_across_seeds(tag=experiment_tag,
                                                      simple_classifier_parameters=simple_classifier_parameters,
                                                      list_of_data_dataframes=[df_train, df_test, df_val],
                                                      data_processing_function=TagsDFWrapper.identity,
                                                      data_processing_function_parameters={},
                                                      normalize=normalize,
                                                      log_wandb=False,
                                                      print_results=True,
                                                      random_seeds=random_seeds
                                                      )
    return average_df


def __get_parser__():
    parser = argparse.ArgumentParser(description="Extract image embeddings using PRETRAINED_MODEL.")
    parser.add_argument("--PRETRAINED_MODEL", type=str, required=True,
                        help="Pretrained model path or name (e.g. openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g. privacyalert, vispr)")
    parser.add_argument("--concept_type", type=str, required=True, help="Name of the concept type "
                                                                        "(e.g. dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot)")
    parser.add_argument("--classifier_config_name", type=str, required=True, help="Name of the config that should be used for classifier training from configs/ "
                                                                        "(e.g. train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_privacyalert)")

    return parser


if __name__ == '__main__':
    args = __get_parser__().parse_args()

    dataset_name = args.dataset_name
    concept_type = args.concept_type
    PRETRAINED_MODEL = args.PRETRAINED_MODEL
    classifier_config_name = args.classifier_config_name

    run_model(PRETRAINED_MODEL, dataset_name, concept_type,
              config_name=classifier_config_name)
    print('Finished training\n\n\n')

    print('Evaluation of the trained model across multiple seeds:')
    _ = evauate_model_with_multiple_seeds(PRETRAINED_MODEL, dataset_name, concept_type,
                                          config_name=classifier_config_name
                                          )
