<!--
 SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>

 SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>

 SPDX-License-Identifier: Apache-2.0
-->

# PrivLEX: Detecting legal concepts in images through Vision-Language Models
**Abstract**. 
We present PrivLEX, a novel image privacy classifier that grounds its decisions in legally defined personal data concepts. PrivLEX is the first interpretable privacy classifier aligned with legal concepts that leverages the recognition capabilities of Vision-Language Models (VLMs). PrivLEX relies on zero-shot VLM concept detection to provide interpretable classification through a label-free Concept Bottleneck Model, without requiring explicit concept labels during training. We demonstrate PrivLEX’s ability to identify personal data concepts that are present in images. We further analyse the sensitivity of such concepts as perceived by human annotators of image privacy datasets.
\
[arxiv:2601.09449](https://arxiv.org/abs/2601.09449)
## Installation
Copy the repository:
```
git clone https://gitlab.idiap.ch/dbaranouskaya/privlex.git
```

Install [conda](https://docs.conda.io/en/latest/) and create a new environment:
```
conda create -n privlex python=3.10  # create the environment, we suggest using python 3.10
conda activate privlex               # activate the environment
```
Go to the project's directory and install all the necessary requirements:
```
cd privlex
pip install -r requirements.txt
```
Set ```PYTHONPATH``` for the project:
```
export PYTHONPATH=$(pwd)
```
## Usage
### Concepts
PrivLEX used [DPV-PD Version 2.0](https://w3c.github.io/dpv/2.0/pd/) [1] (released under [W3C Document license](./LICENSES/LicenseRef-W3C-Software-and-Document-license-2023.txt)) concept that were preprocessed according to the hierarchical structure.
Concepts with their definitions are stored in  ```data/concept/types/dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot.csv```.

### Prepare datasets
PrivLEX code currently supports:
- [PrivacyAlert](https://zenodo.org/records/6406870) [2] (released under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode)) — ```privacyalert```  
- [VISPR](https://tribhuvanesh.github.io/vpa/) [3] — ```vispr```

**Note:** The VISPR dataset is released under [CC BY-NC 4.0 license](https://tribhuvanesh.github.io/vpa/).\
**VISPR preprocessing**\
To use VISPR with PrivLEX:
1. Binarise VISPR annotations according to the presence of the 'safe' attribute.
2. Concatenate train, validation, and test annotations into a single dataframe.
3. Save it as ```data/image_datasets/vispr.csv```.
The dataframe must contain ```'photoid'``` and ```'privacy'``` columns, row indices 0 - 9999, 10000 - 14166, 14167 - 22166 must correspond to train, validation and test subsets.

**Adding a New Dataset**\
To add a dataset ```dataset1```:
1. Create a dataframe with:
   - Two columns named ```'photoid'``` and ```'privacy'```, where the values in ```'privacy'``` equal ```1``` for private images and ```0``` for public. ```'photoid'``` must match image filenames ```{photoid}.jpg```. 
   - Save the dataframe as ```data/image_datasets/dataset1.csv```. 
2. Implement dataset train, validation and test splitting inside a function ```get_dataset_train_test_val_select``` in the ```dataset.py``` file. The ```get_dataset_train_test_val_select``` function must take the ```dataset_name='dataset1'``` and a dataframe and return three dataframes corresponding to train, test and validation split of the input dataframe.

### Specifying paths
In ```utils.py``` define:
1. ```PRIVLEX_PROJECT_PATH``` — project root directory.
2. ```RESULT_PATH``` — directory for trained models and visualisations.
3. Dataset image paths:
   - ```PRIVACYALERT_IMG_PATH``` — directory containing PrivacyAlert .jpg images.
   - ```VISPR_IMG_PATH``` — directory containing ```train2017/```, ```test2017/```, ```val2017/``` folders with .jpg images.
   For VISPR images that are not in .jpg format, convert them to .jpg and save them in the ```images_with_transformed_format/``` folder inside ```VISPR_IMG_PATH```.


### Generate image and concept embeddings
To generate image and concept embeddings we use Huggingface [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip). For more details please refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/clip). 
To generate image embeddings run the following command:
```
python pipeline/vlm_embeddings/extract_image_embeddings.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 [--dataset_name DATASET_NAME]
```
Output directory: ```data/vlm_embeddings/image_embeddings/```.


To generate concept embeddings run the following command:
```
python pipeline/vlm_embeddings/extract_concept_embeddings.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot'
```
Output directory: ```data/vlm_embeddings/concept_embeddings/```.


### Train a privacy classifier on concept scores
To train a privacy classifier on the concept scores with hyperparameters we optimised for the datasets run the following command:\
**PrivacyAlert**
```
python pipeline/train_linear_layer_on_concept_scores/train_linear_layer_on_concept_scores.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --dataset_name privacyalert --classifier_config_name 'train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_privacyalert' --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot'
```

**VISPR**
```
python pipeline/train_linear_layer_on_concept_scores/train_linear_layer_on_concept_scores.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --dataset_name vispr --classifier_config_name 'train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_vispr' --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot'
```
**Hyperparameter optimisation**:\
To select the training hyperparameters we used [Optuna](https://optuna.org/) on 100 steps across the following grid values: 
1. ```C```: from $1e^{−10}$ to $1$ to promote
sparsity
2. ```max_iter```: from $1$ to $250$.

To run the hyperparameter optimisation and train the model with the optimised hyperparameters use: 
```
python pipeline/train_linear_layer_on_concept_scores/train_linear_layer_on_concept_scores.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 [--dataset_name DATASET_NAME] --classifier_config_name train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1] --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot'
```


### Output personal data concepts identified for images
To present personal data concepts identified by PrivLEX for test images run:\
**PrivacyAlert**
```
python pipeline/explainability/generate_explanations_for_images.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --dataset_name privacyalert --classifier_config_name 'train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_privacyalert' --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot' [--list_of_photoids LIST_OF_IMAGE_PHOTOIDS]
```

**VISPR**
```
python pipeline/explainability/generate_explanations_for_images.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --dataset_name vispr --classifier_config_name 'train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_vispr' --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot' [--list_of_photoids LIST_OF_IMAGE_PHOTOIDS]
```

### Visualise the weights of the Logistic Regression
To visualise the weights of the Logistic Regression for both PrivacyAlert and VISPR datasets run:
```
python pipeline/explainability/visualise_lr_weights.py --PRETRAINED_MODEL openai/clip-vit-base-patch32 --dataset_name 'privacyalert, vispr' --classifier_config_names 'train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_privacyalert; train_bipd_v2_logreg-nonbalancedloss-l1score-Cbelow1-norm[0, 1]_vispr' --concept_type 'dpv-pd-v2_with_baseline-dpv-pd-descriptions-and-name-separated-by-colon-no-dot'
```
The script visualises:
- all Logistic Regression weights
- top 20 weights with the highest contribution to private class
- top 20 weights with the highest contribution to public class
- the weights with the highest STD across the datasets.

## Authors and acknowledgment
**Authors:**\
Darya Baranouskaya, [darya.baranouskaya@{idiap.ch, epfl.ch}](darya.baranouskaya@idiap.ch)\
Andrea Cavallaro, [andrea.cavallaro@epfl.ch](andrea.cavallaro@epfl.ch)

**Citation:**
```
@misc{baranouskaya2026privlexdetectinglegalconcepts,
      title={PrivLEX: Detecting legal concepts in images through Vision-Language Models}, 
      author={Darya Baranouskaya and Andrea Cavallaro},
      year={2026},
      eprint={2601.09449},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.09449}, 
}
```

## References
[1] J. Pandit, H., Esteves, B., P. Krog, G., Ryan, P., Golpayegani, D., Flake, J.: Data Privacy Vocabulary (DPV) – version 2.0. In: The Semantic Web – ISWC 2024. pp. 171–193 (2025)\
[2] Zhao, C., Mangat, J., Koujalgi, S., Squicciarini, A., Caragea, C.: Privacyalert: A dataset for image privacy prediction. In: International AAAI Conference on Web and Social Media. vol. 16, pp. 1352–1361 (2022)\
[3] Orekondy, T., Schiele, B., Fritz, M.: Towards a visual privacy advisor: Understanding and predicting privacy risks in images. In: IEEE International Conference on Computer Vision. pp. 3686–3695 (2017)
