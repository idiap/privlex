#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import numpy as np
import random
from torch import manual_seed as torch_set_manual_seed
import matplotlib.pyplot as plt
import torch
import optuna
import optuna.visualization
import pickle

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from metrics import compute_metrics_during_traning
from logger import Logger, SimpleWandbLogger

from dataset import TagsDFWrapper

from tqdm import tqdm
import os

from utils import __get_privlex_project_path__, __get_results_path__

PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'


def normalise_input_data(df_train, df_test, df_val, normalize, return_normalisation_params=False, return_dataframes=False):
    x_train, y_train = df_train.iloc[:, 2:], df_train.iloc[:, 1]
    x_test, y_test = df_test.iloc[:, 2:], df_test.iloc[:, 1]
    x_val, y_val = df_val.iloc[:, 2:], df_val.iloc[:, 1]

    normalisation_params= None

    if normalize == 'min_max_to_0_1':
        df_min, df_max = x_train.min(axis=0), (x_train - x_train.min(axis=0)).max(axis=0)
        df_max[df_max == 0] = 1
        normalisation_params = (df_min, df_max)
        x_train = (x_train - df_min) / df_max
        x_test = (x_test - df_min) / df_max
        x_val = (x_val - df_min) / df_max
    elif normalize=='True' or normalize is True:
        df_mean, df_std = x_train.mean(axis=0), x_train.std(axis=0)
        df_std = df_std + 1e-8
        normalisation_params = (df_mean, df_std)
        x_train = (x_train - df_mean) / df_std
        x_test = (x_test - df_mean) / df_std
        x_val = (x_val - df_mean) / df_std

    if return_dataframes:
        df_train_norm, df_test_norm, df_val_norm = pd.concat([df_train.iloc[:, :2], x_train], axis=1), \
                                                   pd.concat([df_test.iloc[:, :2], x_test], axis=1), \
                                                   pd.concat([df_val.iloc[:, :2], x_val], axis=1)
        if return_normalisation_params:
            return df_train_norm, df_test_norm, df_val_norm, normalisation_params
        return df_train_norm, df_test_norm, df_val_norm

    if return_normalisation_params:
        return (x_train, y_train), (x_test, y_test), (x_val, y_val), normalisation_params
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def run_simple_classifier(tag, df_train, df_test, df_val,
                          simple_classifier_parameters,
                          random_state=1347, print_results=True, normalize=True,
                          return_predictions=False, return_model=False, fit_parameters={}):
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = normalise_input_data(df_train, df_test, df_val, normalize)

    classifier = simple_classifier_parameters.pop('classifier')
    if classifier == 'mlp':
        model = MLPClassifier(random_state=random_state, **simple_classifier_parameters)
    elif classifier == 'logreg':
        model = LogisticRegression(random_state=random_state, **simple_classifier_parameters)
    elif classifier == 'svm':
        model = SVC(random_state=random_state, **simple_classifier_parameters)
    else:
        raise Exception("Classifier model doesn't exist")

    # workaround crutch
    #otherwise when the code of the function is run multiple times,
    # like with different seeds, dictinary key gets removed
    simple_classifier_parameters['classifier'] = classifier

    model.fit(x_train, y_train, **fit_parameters)
    if print_results:
        print(f"{tag}:", model)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_val = model.predict(x_val)
    from logger import Logger

    train_metrics_dict = compute_metrics_during_traning(y_train, y_pred_train)
    test_metrics_dict = compute_metrics_during_traning(y_test, y_pred_test)
    val_metrics_dict = compute_metrics_during_traning(y_val, y_pred_val)

    if print_results:
        print('train')
        Logger.print_metrics_for_binary_cls(train_metrics_dict)

        print('test')
        Logger.print_metrics_for_binary_cls(test_metrics_dict)

        print('val')
        Logger.print_metrics_for_binary_cls(val_metrics_dict)
        print("\n")

    if return_predictions and return_model:
        return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val), model
    elif return_predictions:
        return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val)
    elif return_model:
        return train_metrics_dict, test_metrics_dict, val_metrics_dict, model
    return train_metrics_dict, test_metrics_dict, val_metrics_dict


def eval_simple_classifier(tag, df_train, df_test, df_val,
                           model, print_results=True, normalize=True,
                          return_predictions=False):
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = normalise_input_data(df_train, df_test, df_val, normalize)

    if print_results:
        print(f"{tag}:", model)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_val = model.predict(x_val)
    from logger import Logger

    train_metrics_dict = compute_metrics_during_traning(y_train, y_pred_train)
    test_metrics_dict = compute_metrics_during_traning(y_test, y_pred_test)
    val_metrics_dict = compute_metrics_during_traning(y_val, y_pred_val)

    if print_results:
        print('train')
        Logger.print_metrics_for_binary_cls(train_metrics_dict)

        print('test')
        test_answ = tag
        keys = ['acc', 'ba', 'f1-macro', 'priv_pr', 'priv_rec', 'priv_f1', 'pub_pr', 'pub_rec', 'pub_f1']
        for k in keys:
            test_answ += ' & ' + str(np.round(test_metrics_dict[k], 4))
        test_answ += ' \\\\'
        print(test_answ)
        Logger.print_metrics_for_binary_cls(test_metrics_dict)

        print('val')
        Logger.print_metrics_for_binary_cls(val_metrics_dict)
        print("\n")

    if return_predictions:
        return (train_metrics_dict, test_metrics_dict, val_metrics_dict), (y_pred_train, y_pred_test, y_pred_val)
    return train_metrics_dict, test_metrics_dict, val_metrics_dict


def statistical_comparison_of_classifiers_performance(dict_input_for_cls1, dict_input_for_cls2, n_seeds=550):
    np.random.seed(42)
    seeds = np.random.choice(list(range(5000)), n_seeds, replace=False)
    acc_ba_f1m_statistics = []
    for random_seed in tqdm(seeds):
        train_metrics_dict1, test_metrics_dict1, val_metrics_dict1 = run_simple_classifier(**dict_input_for_cls1, random_state=random_seed)
        train_metrics_dict2, test_metrics_dict2, val_metrics_dict2 = run_simple_classifier(**dict_input_for_cls2, random_state=random_seed)
        curr_metrics = [test_metrics_dict1['acc'], test_metrics_dict2['acc'],
                        test_metrics_dict1['ba'], test_metrics_dict2['ba'],
                        test_metrics_dict1['f1-macro'], test_metrics_dict2['f1-macro']]
        acc_ba_f1m_statistics.append(curr_metrics)
    acc_ba_f1m_statistics_df = pd.DataFrame(np.array(acc_ba_f1m_statistics) * 100, columns=['cls1_acc', 'cls2_acc', 'cls1_ba', 'cls2_ba', 'cls1_f1m', 'cls2_f1m'])
    from scipy.stats import ttest_rel
    print(f'm1: {dict_input_for_cls1["tag"]}')
    print(f"Average acc: m1 {np.mean(acc_ba_f1m_statistics_df['cls1_acc']):.3}({np.std(acc_ba_f1m_statistics_df['cls1_acc']):.3}),"
          f" m2 {np.mean(acc_ba_f1m_statistics_df['cls2_acc']):.3}({np.std(acc_ba_f1m_statistics_df['cls2_acc']):.3})\t"
          f"ba: m1 {np.mean(acc_ba_f1m_statistics_df['cls1_ba']):.3}({np.std(acc_ba_f1m_statistics_df['cls1_ba']):.3}),"
          f" m2 {np.mean(acc_ba_f1m_statistics_df['cls2_ba']):.3}({np.std(acc_ba_f1m_statistics_df['cls2_ba']):.3}) \t"
          f"f1-macro: m1 {np.mean(acc_ba_f1m_statistics_df['cls1_f1m']):.3}({np.std(acc_ba_f1m_statistics_df['cls1_f1m']):.3}),"
          f" m2 {np.mean(acc_ba_f1m_statistics_df['cls2_f1m']):.3}({np.std(acc_ba_f1m_statistics_df['cls2_f1m']):.3})\n")

    t_stat_acc, p_value_acc = ttest_rel(acc_ba_f1m_statistics_df['cls1_acc'], acc_ba_f1m_statistics_df['cls2_acc'])
    print(f"ACC T-statistic: {t_stat_acc}, p-value: {p_value_acc}, one model is better:{p_value_acc < 0.05} ")

    t_stat_ba, p_value_ba = ttest_rel(acc_ba_f1m_statistics_df['cls1_ba'], acc_ba_f1m_statistics_df['cls2_ba'])
    print(f"BA T-statistic: {t_stat_ba}, p-value: {p_value_ba}, one model is better:{p_value_ba < 0.05} ")

    t_stat_f1m, p_value_f1m = ttest_rel(acc_ba_f1m_statistics_df['cls1_f1m'], acc_ba_f1m_statistics_df['cls2_f1m'])
    print(f"F1-macro T-statistic: {t_stat_ba}, p-value: {p_value_f1m}, one model is better:{p_value_f1m < 0.05} ")
    return acc_ba_f1m_statistics_df


def run_classifier_training_across_seeds(tag,
                                         simple_classifier_parameters,
                                         list_of_data_dataframes,
                                         data_processing_function=TagsDFWrapper.identity,
                                         data_processing_function_parameters={},
                                         normalize=True,
                                         print_results=True, log_wandb=True,
                                         random_seeds=[1347, 20, 4678, 829, 1558, 39, 991, 96, 4, 435], fit_parameters={}):
    '''
    :param tag: name of the experiment
    :param model_train_eval_function:
    :param model_train_eval_function_parameters:
    :param list_of_data_dataframes: list of train, test and eval dataframes,
           where each df has the first column photoid and the second columns privacy
    :param data_processing_function: func to process dataframeis some way
    :param data_processing_function_parameters: params the previous function takes
    :param print_results:
    :param log_wandb:
    :param random_seeds:
    :return:
    '''
    keys = ['acc', 'ba', 'f1-macro', 'priv_pr', 'priv_rec', 'priv_f1', 'pub_pr', 'pub_rec', 'pub_f1']
    dfs = []

    for i in range(len(random_seeds)):
        random_state = random_seeds[i]

        np.random.seed(random_state)
        random.seed(random_state)
        torch_set_manual_seed(random_state)

        df_train, df_test, df_val = data_processing_function(list_of_data_dataframes, **data_processing_function_parameters)
        train, test, val = run_simple_classifier(tag=tag,
                                                 df_train=df_train, df_test=df_test, df_val=df_val,
                                                 simple_classifier_parameters=simple_classifier_parameters,
                                                 random_state=random_state,
                                                 normalize=normalize,
                                                 print_results=False, fit_parameters=fit_parameters)
        dfs.append(pd.DataFrame([['train', random_state] + [train[k] * 100 for k in keys],
                                 ['test', random_state] + [test[k] * 100 for k in keys],
                                 ['val', random_state] + [val[k] * 100 for k in keys]],
                                columns=['dataset', 'random_state'] + keys))
    metrics_cols = keys
    mean_df_numpy = np.mean([df[metrics_cols].to_numpy() for df in dfs], axis=0)
    std_df_numpy = np.std([df[metrics_cols].to_numpy() for df in dfs], axis=0)
    average_df_numpy = np.concatenate([np.expand_dims(mean_df_numpy, 2), np.expand_dims(std_df_numpy, 2)], axis=2)
    average_df_numpy = average_df_numpy.reshape(
        (average_df_numpy.shape[0], average_df_numpy.shape[1] * average_df_numpy.shape[2]))
    average_df = pd.concat([pd.DataFrame(np.array([['train', 'test', 'val']]).T, columns=['dataset']),
                            pd.DataFrame(average_df_numpy,
                                         columns=np.array([[metric_name + ' mean', metric_name + ' std']
                                                           for metric_name in metrics_cols]).flatten())], axis=1)
    if print_results:
        print('\n tag')
        print('\n Performance Across Multiple Seeds')
        Logger.print_statistics_from_dataframe(average_df, '', '', run_name=tag)

    if log_wandb:
        logger = SimpleWandbLogger(project='privlex', name=tag,
                             config=simple_classifier_parameters)
        logger.log_dataframe('average runs across 10 seeds', average_df)
    return average_df


def run_classifier_training_across_seeds_and_save_weights(tag, results_dir,
                                         simple_classifier_parameters,
                                         list_of_data_dataframes,
                                         data_processing_function=TagsDFWrapper.identity,
                                         data_processing_function_parameters={},
                                         normalize=True,
                                         print_results=True, log_wandb=True,
                                         random_seeds=[1347, 20, 4678, 829, 1558, 39, 991, 96, 4, 435], fit_parameters={}):
    '''
    :param tag: name of the experiment
    :param model_train_eval_function:
    :param model_train_eval_function_parameters:
    :param list_of_data_dataframes: list of train, test and eval dataframes,
           where each df has the first column photoid and the second columns privacy
    :param data_processing_function: func to process dataframeis some way
    :param data_processing_function_parameters: params the previous function takes
    :param print_results:
    :param log_wandb:
    :param random_seeds:
    :return:
    '''
    keys = ['acc', 'ba', 'f1-macro', 'priv_pr', 'priv_rec', 'priv_f1', 'pub_pr', 'pub_rec', 'pub_f1']
    dfs = []

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for i in range(len(random_seeds)):
        random_state = random_seeds[i]

        np.random.seed(random_state)
        random.seed(random_state)
        torch_set_manual_seed(random_state)

        df_train, df_test, df_val = data_processing_function(list_of_data_dataframes, **data_processing_function_parameters)
        train, test, val, model = run_simple_classifier(tag=tag,
                                                 df_train=df_train, df_test=df_test, df_val=df_val,
                                                 simple_classifier_parameters=simple_classifier_parameters,
                                                 random_state=random_state,
                                                 normalize=normalize,
                                                 print_results=False, fit_parameters=fit_parameters, return_model=True)
        dfs.append(pd.DataFrame([['train', random_state] + [train[k] * 100 for k in keys],
                                 ['test', random_state] + [test[k] * 100 for k in keys],
                                 ['val', random_state] + [val[k] * 100 for k in keys]],
                                columns=['dataset', 'random_state'] + keys))

        filename = results_dir + f'{tag}_seed-{random_state}.sav'
        pickle.dump(model, open(filename, 'wb'))

    metrics_cols = keys
    mean_df_numpy = np.mean([df[metrics_cols].to_numpy() for df in dfs], axis=0)
    std_df_numpy = np.std([df[metrics_cols].to_numpy() for df in dfs], axis=0)
    average_df_numpy = np.concatenate([np.expand_dims(mean_df_numpy, 2), np.expand_dims(std_df_numpy, 2)], axis=2)
    average_df_numpy = average_df_numpy.reshape(
        (average_df_numpy.shape[0], average_df_numpy.shape[1] * average_df_numpy.shape[2]))
    average_df = pd.concat([pd.DataFrame(np.array([['train', 'test', 'val']]).T, columns=['dataset']),
                            pd.DataFrame(average_df_numpy,
                                         columns=np.array([[metric_name + ' mean', metric_name + ' std']
                                                           for metric_name in metrics_cols]).flatten())], axis=1)
    if print_results:
        print('\n tag')
        print('\n AVERAGE RESULTS')
        Logger.print_statistics_from_dataframe(average_df, '', '', run_name=tag)

    if log_wandb:
        logger = SimpleWandbLogger(project='2023-04_simple_classifiers', name=tag,
                             config=simple_classifier_parameters)
        logger.log_dataframe('average runs across 10 seeds', average_df)
    return average_df



def optimize_hyperparameters_with_optuna(set_to_optimize_on, metric_to_optimize,
                                         parameters_to_optimize,
                                         other_simple_classifier_parameters,
                                         tag,
                                         df_train, df_test, df_val,
                                         random_state, normalize=True,
                                         n_trials=100, timeout=500,
                                         print_results=True, fit_parameters={}):
    '''
    :param set_to_optimize_on:  'val' (correct) or 'test' (cheating)
    :param metric_to_optimize: one of possibel metrics - 'acc', 'ba', 'f1-macro', 'priv_pr', 'priv_rec', 'priv_f1', 'pub_pr', 'pub_rec', 'pub_f1'
    :param trial:
    :param parameters_to_optimize:
        {
        'lr': (min, max, log=False)
        }
    :param other_simple_classifier_parameters:
    :return:
    '''
    def objective_function(trial):

        def init_parameter_suggest_trial_from_tuple_of_values(param_name, param_tuple):
            '''
            :param param_tuple:  (type, (min, max), step, log) ('int', (0, 100), step, True)
                                low and high values are both included in the range
            '''
            param_type, param_values_dict = param_tuple
            if param_type == 'categorical':
                return trial.suggest_categorical(param_name, **param_values_dict)
            elif param_type == 'float':
                assert param_values_dict['low']  < param_values_dict['high']
                assert ('log' in param_values_dict and type(param_values_dict['log']) == bool) or ('log' not in param_values_dict)
                return trial.suggest_float(param_name, **param_values_dict)

            elif param_type == 'int':
                assert param_values_dict['low'] < param_values_dict['high']
                assert ('log' in param_values_dict and type(param_values_dict['log']) == bool) or ('log' not in param_values_dict)
                return trial.suggest_int(param_name, **param_values_dict)
            else:
                raise Exception('Unknown type')

        simple_classifier_parameters = {}
        for k in other_simple_classifier_parameters:
            simple_classifier_parameters[k] = other_simple_classifier_parameters[k]
        for k in parameters_to_optimize:
            simple_classifier_parameters[k] = init_parameter_suggest_trial_from_tuple_of_values(k, parameters_to_optimize[k])

        if 'classifier' in other_simple_classifier_parameters:
            train, test, val = run_simple_classifier(tag=tag,
                                                     df_train=df_train, df_test=df_test, df_val=df_val,
                                                     simple_classifier_parameters=simple_classifier_parameters,
                                                     random_state=random_state,
                                                     normalize=normalize,
                                                     print_results=False, fit_parameters=fit_parameters)
        else:
            raise Exception('task is not implemented')
        # print(test)
        if set_to_optimize_on == 'val':
            return val[metric_to_optimize]
        elif set_to_optimize_on == 'test':
            return test[metric_to_optimize]
        else:
            raise Exception('Unknown set to optimize on')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_function, n_trials=n_trials, timeout=timeout)

    print("Best trial")
    print("Value: ", study.best_trial.value)

    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))

    if print_results:
        simple_classifier_parameters = {}
        for k in other_simple_classifier_parameters:
            simple_classifier_parameters[k] = other_simple_classifier_parameters[k]
        for k in parameters_to_optimize:
            simple_classifier_parameters[k] = study.best_trial.params[k]
        # print(simple_regressor_parameters)

        print()
        print('optuna\_best\_trial (', end='')
        for k in simple_classifier_parameters:
            print(k.replace('_', '\_'), '=', simple_classifier_parameters[k], sep='', end=', ')
        print(') \\\\')

        if 'classifier' in other_simple_classifier_parameters:
            average_df = run_classifier_training_across_seeds(tag=tag + '_optuna_best_trial',
                                                              simple_classifier_parameters=simple_classifier_parameters,
                                                              list_of_data_dataframes=[df_train, df_test, df_val],
                                                              data_processing_function=TagsDFWrapper.identity,
                                                              data_processing_function_parameters={},
                                                              normalize=normalize,
                                                              print_results=print_results, log_wandb=False,
                                                              random_seeds=[1347, 20, 4678, 829, 1558, 39, 991, 96, 4, 435], fit_parameters=fit_parameters)
        else:
            raise Exception('task is not implemented')



        return study.best_trial, average_df


    return study.best_trial


