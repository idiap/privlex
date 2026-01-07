#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import torch
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from pipeline.backbone_model.extract_features import extract_features
from utils import __get_privlex_project_path__
from pipeline.backbone_model import get_vlm_model_wrapper_by_name


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'


def extract_concept_features_with_huggingface_tranformer(PRETRAINED_MODEL, df, results_path):

    model_wrapper = get_vlm_model_wrapper_by_name(PRETRAINED_MODEL)
    model_wrapper = model_wrapper(PRETRAINED_MODEL,
                                  init_text_encoder=True,
                                  init_vision_encoder=False, init_full_model=False,
                                  init_processor=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_wrapper.to(device)


    print('Model:', model_wrapper.text_model)
    print(f'max_lengh: {model_wrapper.max_length},\n Processor:', model_wrapper.processor)

    # create dataloader
    from dataset import CustomCLIPDataset
    dataset = CustomCLIPDataset(df, model_wrapper.processor,
                                input_text=True, input_image=False, return_photoid=True, max_length=model_wrapper.max_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)


    if PRETRAINED_MODEL.split('/')[-1] not in results_path:
        raise Exception("Results path doesn't match the model.")
    print('Results path:', results_path)

    def forward_call(model_wrapper, data):
        return model_wrapper.forward_call_text(data)

    results_df = extract_features(dataloader, model_wrapper, forward_call, results_path)

    results_df.to_csv(results_path, index=False)
    return results_df



def extract_features_with_huggingface_tranformer_from_terms_descriptions(PRETRAINED_MODEL,
                                                                         terms,
                                                                         terms_descriptions,
                                                                         terms_text,
                                                                         results_path):

    df = pd.DataFrame([], columns=[])
    df['photoid'] = list(range(len(terms)))
    df['privacy'] = list(range(len(terms)))
    df['text'] = terms_text

    extract_concept_features_with_huggingface_tranformer(PRETRAINED_MODEL, df, results_path)

    results_df = pd.read_csv(results_path)
    results_df = results_df.drop(columns=['privacy', 'photoid'])

    if terms_descriptions is None:
        raise Exception("terms_descriptions is None, this is depricated in the current version of the algorithm")
    else:
        results_df = pd.concat([pd.DataFrame(np.array([terms, terms_descriptions, terms_text]).T, columns=['Term', 'Description', 'text']),
                                results_df], axis=1)
    results_df.to_csv(results_path, index=False)
    return results_df


def __get_parser__():
    parser = argparse.ArgumentParser(description="Extract concept embeddings using PRETRAINED_MODEL.")
    parser.add_argument("--PRETRAINED_MODEL", type=str, required=True,
                        help="Pretrained model path or name (e.g. openai/clip-vit-base-patch32)")
    parser.add_argument("--concept_type", type=str, required=True, help="Name of the concept type "
                                                                        "(e.g. dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot)")

    return parser


if __name__ == '__main__':
    args = __get_parser__().parse_args()

    concept_type = args.concept_type
    PRETRAINED_MODEL = args.PRETRAINED_MODEL


    results_path = data_path + '/vlm_embeddings/concept_embeddings/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_path = results_path + f'{concept_type}__concept-embeddings__with_{PRETRAINED_MODEL.split("/")[1]}_model.csv'

    concepts_df = pd.read_csv( data_path + f'concept_types/{concept_type}.csv')
    terms, terms_descriptions, terms_text = concepts_df['Term'], concepts_df['Description'], concepts_df['text']

    results_df = extract_features_with_huggingface_tranformer_from_terms_descriptions(PRETRAINED_MODEL,
                                                                         terms,
                                                                         terms_descriptions,
                                                                         terms_text,
                                                                         results_path)


