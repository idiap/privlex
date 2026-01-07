#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from tqdm import tqdm


def extract_features(dataloader, model, forward_call, results_path, model_inner_reps=None, columns=None):
    if model_inner_reps is None:
        model_inner_reps = []
    model.eval()
    def save_df():
        cols = columns
        if cols is None:
            cols = list(range(len(model_inner_reps[0]) - 2))
        model_inner_reps_df = pd.DataFrame(model_inner_reps,
                                           columns=['photoid', 'privacy'] + cols)
        model_inner_reps_df.to_csv(results_path, index=False)
        return model_inner_reps_df

    sz = 0
    for data in tqdm(dataloader):
        photoids = data.pop('photoid')
        labels = data.pop('target')

        try:
            photoids = [int(phid) for phid in photoids.numpy()]
        except AttributeError:
            print('photoid is a string')

        labels = labels.numpy()

        sz += len(labels)
        if len(model_inner_reps) >= sz:
            continue

        output = forward_call(model, data)


        model_inner_reps.extend([[photoids[i]] + [int(labels[i])]+ list(output[i]) for i in range(output.shape[0])])
        if len(model_inner_reps) // 300 >  (len(model_inner_reps) - len(output)) // 300:
            _ = save_df()
    model_inner_reps_df = save_df()
    return model_inner_reps_df
