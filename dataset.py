#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import torch
from utils import find_img_path_by_photoid
from torch.utils.data import Dataset
from PIL import Image


class CustomCLIPDataset(Dataset):
    def __init__(self, dataframe, processor,
                 input_text, input_image,
                 return_photoid=False, max_length=77, non_int_photoids=False):
        self.dataframe = dataframe
        self.processor = processor
        self.input_text = input_text
        self.input_image = input_image
        self.return_photoid = return_photoid
        self.max_length = max_length
        self.non_int_photoids = non_int_photoids

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]
        photoid, privacy = row['photoid'], row['privacy']

        # Load image using the find_img_path_by_photoid function
        image = None
        text = None
        if self.input_image:
            image = Image.open(find_img_path_by_photoid(photoid)).convert("RGB")
        if self.input_text:
            text = row['text']

        # Process text using the CLIP tokenizer
        inputs = self.processor(text=text, images=image,
                                return_tensors="pt", padding='max_length', max_length=self.max_length,
                                truncation=True)
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)

        inputs['target'] = torch.Tensor([privacy])
        if self.return_photoid:
            if self.non_int_photoids:
                inputs['photoid'] = torch.LongTensor([idx])
            else:
                inputs['photoid'] = torch.LongTensor([photoid])

        # Return processed inputs and labels
        return inputs


def privacyalert_train_test_val_select(df):
    pra_train, pra_test, pra_val = df.iloc[:3136], df.iloc[5000:], df.iloc[3136:5000]
    pra_train.reset_index(inplace=True, drop=True)
    pra_test.reset_index(inplace=True, drop=True)
    pra_val.reset_index(inplace=True, drop=True)
    return pra_train, pra_test, pra_val


def vispr_train_test_val_select(df):
    vispr_train, vispr_test, vispr_val = df.iloc[:10000], df.iloc[14167:], df.iloc[10000:14167]
    vispr_train.reset_index(inplace=True, drop=True)
    vispr_test.reset_index(inplace=True, drop=True)
    vispr_val.reset_index(inplace=True, drop=True)
    return vispr_train, vispr_test, vispr_val


def get_dataset_train_test_val_select(dataset_name, df, **kwargs):
    if dataset_name == 'privacyalert' or dataset_name == 'pra':
        return privacyalert_train_test_val_select(df)
    elif dataset_name == 'vispr':
        return vispr_train_test_val_select(df)
    raise Exception('Unknown dataset')


class DataFrameWrapper():
    def __init__(self, df):
        self.df = df


class TagsDFWrapper():
    @staticmethod
    def identity(list_of_data_dataframes):
        return list_of_data_dataframes
