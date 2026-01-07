#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoProcessor



class VLMModelHuggingFaceWrapper(nn.Module):
    def __init__(self, PRETRAINED_MODEL,
                 init_text_encoder=False, init_vision_encoder=False, init_full_model=False,
                 init_processor=False):
        super().__init__()
        self.PRETRAINED_MODEL = PRETRAINED_MODEL

        if init_text_encoder:
            self.__init_text_model__()

        if init_vision_encoder:
            self.__init_vision_model__()

        if init_full_model:
            self.__init_full_model__()

        if init_processor:
            self.__init__processor__()

    def __init_vision_model__(self):
        pass

    def __init_text_model__(self):
        pass

    def __init_full_model__(self):
        self.full_model = AutoModel.from_pretrained(self.PRETRAINED_MODEL)

    def __init__processor__(self):
        self.processor = AutoProcessor.from_pretrained(self.PRETRAINED_MODEL)

    def forward_call_image(self, data):
        pass

    def forward_call_text(self, data):
        pass

    def compute_cosine_similarity(self, image_embeds, concept_embeds):
        pass


class CLIPModelHuggingFaceWrapper(VLMModelHuggingFaceWrapper):
    def __init__(self, PRETRAINED_MODEL,
                 init_text_encoder=False, init_vision_encoder=False, init_full_model=False,
                 init_processor=False):
        super().__init__(PRETRAINED_MODEL,
                         init_text_encoder=init_text_encoder,
                         init_vision_encoder=init_vision_encoder,
                         init_full_model=init_full_model,
                         init_processor=init_processor)

    def __init_vision_model__(self):
        from transformers import CLIPVisionModelWithProjection
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.PRETRAINED_MODEL)

    def __init_text_model__(self):
        from transformers import CLIPTextModelWithProjection
        self.text_model = CLIPTextModelWithProjection.from_pretrained(self.PRETRAINED_MODEL)
        self.max_length = self.text_model.text_model.embeddings.position_embedding.weight.shape[0]

    def __init__processor__(self):
        from transformers import CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(self.PRETRAINED_MODEL)

    def __init_full_model__(self):
        from transformers import CLIPModel
        self.full_model = CLIPModel.from_pretrained(self.PRETRAINED_MODEL)

    def forward_call_image(self, data):
        device = self.vision_model.device
        assert len(data) == 1
        output = self.vision_model(pixel_values=data['pixel_values'].to(device)).image_embeds
        output = output.detach().cpu().numpy()
        return output

    def forward_call_text(self, data):

        device = self.text_model.device

        assert len(data) == 2
        output = self.text_model(input_ids=data['input_ids'].to(device),
                       attention_mask=data['attention_mask'].to(device)).text_embeds
        output = output.detach().cpu().numpy()
        return output

    def compute_cosine_similarity(self, image_embeds, concept_embeds):
        image_embeddings = image_embeds / np.expand_dims(np.linalg.norm(image_embeds, axis=1), 1)
        concept_embeddings = concept_embeds / np.expand_dims(np.linalg.norm(concept_embeds, axis=1), 1)
        concept_scores = image_embeddings @ concept_embeddings.T
        return concept_scores


def get_vlm_model_wrapper_by_name(model_name):
    if model_name.startswith('clip-vit-') or model_name.startswith('openai/clip-vit-'):
        return CLIPModelHuggingFaceWrapper
    else:
        raise Exception('Unknown model name.')
