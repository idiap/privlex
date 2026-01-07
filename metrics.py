#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(y_train, y_pred_train, y_test, y_pred_test):
    acc_train, acc_test = accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)
    ba_train, ba_test = balanced_accuracy_score(y_train, y_pred_train), balanced_accuracy_score(y_test, y_pred_test)
    f1_train, f1_test = f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)
    f1_macro_train, f1_macro_test = f1_score(y_train, y_pred_train, average='macro'), f1_score(y_test, y_pred_test, average='macro')
    
    print(f'train: acc {acc_train*100:.4},\t ba {ba_train*100:.4}, \t f1 {f1_train*100:.4}, \t f1_macro {f1_macro_train*100:.4} \n'
          f'test: acc {acc_test*100:.4},\t ba {ba_test*100:.4}, \t f1 {f1_test*100:.4}, \t f1_macro {f1_macro_test*100:.4}')
    return (acc_train, acc_test), (ba_train, ba_test), (f1_train, f1_test), (f1_macro_train, f1_macro_test)


def compute_metrics_during_traning(labels, preds):
    '''
    Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
    preds = np.argmax(preds, axis = 1).flatten()
    labels = labels.flatten()
    '''
    metrics_res = {'acc':  accuracy_score(labels, preds),
                   'ba':  balanced_accuracy_score(labels, preds),
                   'f1-macro': f1_score(labels, preds, average='macro'),
                   'priv_pr': precision_score(labels, preds),
                   'priv_rec': recall_score(labels, preds),
                   'priv_f1':  f1_score(labels, preds),
                   'pub_pr': precision_score(1 - np.array(labels), 1 - np.array(preds)),
                   'pub_rec': recall_score(1 - np.array(labels), 1 - np.array(preds)),
                   'pub_f1':  f1_score(1 - np.array(labels), 1 - np.array(preds)),
                  }
    return metrics_res
