#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import torch
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from pipeline.backbone_model.extract_features import extract_features
from utils import __get_privlex_project_path__
from pipeline.backbone_model import get_vlm_model_wrapper_by_name


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'


def extract_image_features_with_huggingface_tranformer(PRETRAINED_MODEL, df, results_path, non_int_photoids):

    model_wrapper = get_vlm_model_wrapper_by_name(PRETRAINED_MODEL)
    model_wrapper = model_wrapper(PRETRAINED_MODEL,
                                  init_text_encoder=False,
                                  init_vision_encoder=True, init_full_model=False,
                                  init_processor=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_wrapper.to(device)


    print('Model:', model_wrapper.vision_model)

    # create dataloader
    from dataset import CustomCLIPDataset
    dataset = CustomCLIPDataset(df, model_wrapper.processor,
                                input_text=False, input_image=True, return_photoid=True, max_length=None, non_int_photoids=non_int_photoids) #, non_int_photoids=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)


    if PRETRAINED_MODEL.split('/')[-1] not in results_path:
        raise Exception("Results path doesn't match the model.")
    print('Results path:', results_path)

    def forward_call(model_wrapper, data):
        return model_wrapper.forward_call_image(data)

    results_df = extract_features(dataloader, model_wrapper, forward_call, results_path)
    if non_int_photoids:
        results_df['photoid'] = df['photoid']
    results_df.to_csv(results_path, index=False)
    return results_df


def __get_parser__():
    parser = argparse.ArgumentParser(description="Extract image embeddings using PRETRAINED_MODEL.")
    parser.add_argument("--PRETRAINED_MODEL", type=str, required=True,
                        help="Pretrained model path or name (e.g. openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g. privacyalert, vispr)")

    return parser


if __name__ == '__main__':
    args = __get_parser__().parse_args()

    dataset_name = args.dataset_name
    PRETRAINED_MODEL = args.PRETRAINED_MODEL

    results_path = data_path + 'vlm_embeddings/image_embeddings/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_path = results_path + f'{dataset_name}__image-embeddings__with_{PRETRAINED_MODEL.split("/")[-1]}_model.csv'

    dataset_df = pd.read_csv(data_path + f'image_datasets/{dataset_name}.csv')
    non_int_photoids = True if type(dataset_df['photoid'][0]) == str else False

    image_emebddings_df = extract_image_features_with_huggingface_tranformer(PRETRAINED_MODEL, dataset_df, results_path, non_int_photoids)
