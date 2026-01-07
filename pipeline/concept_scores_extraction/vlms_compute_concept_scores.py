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
from pipeline.backbone_model import get_vlm_model_wrapper_by_name
from utils import __get_privlex_project_path__


PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'


def extract_concept_scores_for_vlm_embeds(image_embeds, concept_embeds, PRETRAINED_MODEL):
    model_wrapper = get_vlm_model_wrapper_by_name(PRETRAINED_MODEL)
    model_wrapper = model_wrapper(PRETRAINED_MODEL,
                                  init_text_encoder=False,
                                  init_vision_encoder=True, init_full_model=False,
                                  init_processor=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_wrapper.to(device)

    cos_sim = model_wrapper.compute_cosine_similarity(image_embeds, concept_embeds)
    return cos_sim


def extract_concept_scores_for_vlm_embeds_dfs(image_embeds_df, concept_embeds_df, PRETRAINED_MODEL):

    # check the size of embeddings
    assert (image_embeds_df.shape[1] + 1) == concept_embeds_df.shape[1]
    embeds_dim = image_embeds_df.shape[1] - 2


    # check that dataframes save emebddings in the correct format
    assert list(image_embeds_df.columns[:2]) == ['photoid', 'privacy']
    assert list(concept_embeds_df.columns[:3]) == ['Term', 'Description', 'text']

    assert list(image_embeds_df.columns[2:]) == [str(i) for i in range(embeds_dim)]
    assert list(concept_embeds_df.columns[3:]) == [str(i) for i in range(embeds_dim)]

    image_embeds = image_embeds_df.iloc[:, 2:].to_numpy()
    concept_embeds = concept_embeds_df.iloc[:, 3:].to_numpy()

    cos_sim = extract_concept_scores_for_vlm_embeds(image_embeds, concept_embeds, PRETRAINED_MODEL)

    return pd.concat([image_embeds_df.iloc[:, :2], pd.DataFrame(cos_sim, columns=list(concept_embeds_df.Term))], axis=1)


def extract_vlm_concept_scores_by_dataset_concepts_and_model(dataset_name, concept_type, PRETRAINED_MODEL):
    image_embeds_df = pd.read_csv(data_path + f'vlm_embeddings/image_embeddings/{dataset_name}__image-embeddings__with_{PRETRAINED_MODEL.split("/")[-1]}_model.csv' )
    concept_embeds_df = pd.read_csv(data_path + f'vlm_embeddings/concept_embeddings/{concept_type}__concept-embeddings__with_{PRETRAINED_MODEL.split("/")[1]}_model.csv' )

    cos_sim_df = extract_concept_scores_for_vlm_embeds_dfs(image_embeds_df, concept_embeds_df, PRETRAINED_MODEL)
    assert cos_sim_df.shape[0] == image_embeds_df.shape[0]
    assert cos_sim_df.columns[0] == 'photoid' and cos_sim_df.columns[1] == 'privacy'
    return cos_sim_df


