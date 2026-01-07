#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
from pipeline.concept_scores_extraction.vlms_compute_concept_scores import extract_vlm_concept_scores_by_dataset_concepts_and_model
from pipeline.backbone_model import list_of_pre_trained_models
from utils import __get_privlex_project_path__

PRIVLEX_PROJECT_PATH = __get_privlex_project_path__()
data_path = PRIVLEX_PROJECT_PATH + 'data/'


def extract_concept_scores(PRETRAINED_MODEL, dataset_name, concept_type):
    if PRETRAINED_MODEL in list_of_pre_trained_models['vlm']:
        concept_scores_df = extract_vlm_concept_scores_by_dataset_concepts_and_model(dataset_name, concept_type, PRETRAINED_MODEL)
    else:
        raise ValueError('PRETRAINED_MODEL model not recognized')
    return concept_scores_df

