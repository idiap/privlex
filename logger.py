#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
from datetime import date
import re
import numpy as np


class Logger():
    @staticmethod
    def print_metics(metrics_res):
        print(', '.join([ f"{key}: {val* 100:.4}" for key, val in metrics_res.items()]))

    @staticmethod
    def print_metrics_for_overleaf(model_name, dataset_type, metrics_res, metrics_to_print=None):
        if metrics_to_print is None:
            metrics_to_print = list(metrics_res.keys())
        print(f'{model_name} {dataset_type} & ', end='')
        for k in metrics_to_print[:-1]:
            print(f'{metrics_res[k]:.3}', end=' & ')
        print(f'{metrics_res[metrics_to_print[-1]]:.3}', ' \\\\ ')
        return

    @staticmethod
    def print_metrics_for_whole_dataset_for_overleaf(model_name, train_metrics_dict, test_metrics_dict, val_metrics_dict, metrics_to_print=None):
        if metrics_to_print is None:
            metrics_to_print = list(train_metrics_dict.keys())
        print('Model & ' + ' & '.join(metrics_to_print) + ' \\\\')
        Logger.print_metrics_for_overleaf(model_name, 'train', train_metrics_dict, metrics_to_print=metrics_to_print)
        Logger.print_metrics_for_overleaf(model_name, 'test', test_metrics_dict, metrics_to_print=metrics_to_print)
        Logger.print_metrics_for_overleaf(model_name, 'val', val_metrics_dict, metrics_to_print=metrics_to_print)

    @staticmethod
    def print_metrics_for_binary_cls(metrics_res):
        overall_keys = [key for key in metrics_res.keys() if not key.startswith('priv') and not key.startswith('pub')]
        priv_keys = [key for key in metrics_res.keys() if key.startswith('priv')]
        pub_keys = [key for key in metrics_res.keys() if key.startswith('pub')]
        print(', '.join([ f"{key}: {metrics_res[key]* 100:.4}" for key in overall_keys]), end=' || ')
        print(', '.join([ f"{key}: {metrics_res[key]* 100:.4}" for key in priv_keys]), end=' | ')
        print(', '.join([ f"{key}: {metrics_res[key]* 100:.4}" for key in pub_keys]))
        return

    @staticmethod
    def print_metrics_for_regression(metrics_res):
        overall_keys = [key for key in metrics_res.keys()]# if not key.startswith('priv') and not key.startswith('pub')]
        print(', '.join([f"{key}: {metrics_res[key] * 100:.4}" for key in overall_keys]))
        return

    @staticmethod
    def extract_metrics_from_training_output_file(line):
        """Extract metrics from a line and return as a dictionary."""
        metrics = {}
        for part in re.split(r'\|\|? ', line):
            for metrics_line in part.split(','):
                key, value = metrics_line.split(':')
                metrics[key.strip()] = float(value)
        return metrics

    @staticmethod
    def print_statistics_from_dataframe(df, filename, wandb_link, run_name=None):
        if 'epoch' in df.columns:
            epoch_text = ' ep' + str(df['epoch'].to_numpy()[0])
            metrics_cols = df.drop(columns=['dataset', 'epoch']).columns
        else:
            epoch_text = ''
            metrics_cols = df.drop(columns=['dataset']).columns
        print('Model & ' + ' & '.join(list(metrics_cols)) + ' \\\\')
        if run_name is None:
            run_name = filename
            if filename.endswith('.o'):
                run_name = filename.split('.')[0]
        wandb_link_string = f' \\href{{{wandb_link}}}' + '{link} ' if len(wandb_link) > 0 else ''
        line_start = run_name + wandb_link_string + epoch_text
        for i in range(len(df)):
            dataset = df.iloc[i]['dataset']
            metrics = map(lambda x: str(np.round(x, 3)), list(df.iloc[i][metrics_cols]))
            print(line_start + f' {dataset} & ' + ' & '.join(metrics) + ' \\\\')


class SimpleWandbLogger(Logger):
    def __init__(self, project, name, config):
        self.project = project
        self.config = config
        if 'date' in config and config['date'] is not None:
            project_date = config['date']
        else:
            project_date = str(date.today())
        self.name = project_date + '_' + name

        self.initialised = False

    def init_run(self):
        self.wandb_run = wandb.init(project=self.project, name=self.name, config=self.config)
        self.initialised = True

    def log_dataframe(self, tag, df):
        if not self.initialised:
            self.init_run()
        wandb_table = wandb.Table(data=df.to_numpy(), columns=list(df.columns))
        wandb.log({tag: wandb_table}, commit=True)

    def __del__(self):
        try:
            self.wandb_run.finish()
        except:
            print('Wandb run was not initialised')
