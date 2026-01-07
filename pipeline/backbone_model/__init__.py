#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
from .vlm_model_wrapper import get_vlm_model_wrapper_by_name
from .vlm_model_wrapper import CLIPModelHuggingFaceWrapper

list_of_pre_trained_models =  {
    'vlm': ['openai/clip-vit-base-patch32']
}
