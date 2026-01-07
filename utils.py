#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Darya Baranouckaya <darya.baranouskaya@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
import os
from PIL import Image
import json


PRIVLEX_PROJECT_PATH = ...
PRIVACYALERT_IMG_PATH = ...
VISPR_IMG_PATH = ...
RESULT_PATH = ...


def __get_privlex_project_path__():
    return PRIVLEX_PROJECT_PATH

def __get_results_path__():
    return RESULT_PATH


def find_img_path_by_photoid(photoid, privacyalert_path=PRIVACYALERT_IMG_PATH,
                             vispr_path=VISPR_IMG_PATH):
    if type(photoid) != str:
        photoid = int(photoid)
    photoid = str(photoid)
    privacyalert_photo_path = privacyalert_path + '{photoid}.jpg'
    vispr_photo_paths = [vispr_path + f'{tag}2017/' + '{photoid}.jpg' for tag in ['train', 'val', 'test']]
    vispr_photo_paths_for_images_with_transformed_format = [VISPR_IMG_PATH + 'images_with_transformed_format/{photoid}.jpg']


    paths = [privacyalert_photo_path] + vispr_photo_paths + vispr_photo_paths_for_images_with_transformed_format
    for p in paths:
        path = p.format(photoid=photoid)
        if os.path.exists(path):
            return path
    raise Exception(f"Not found file {photoid}")


def convert_image_to_jpeg(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert the image to RGB mode if it's not already, since JPEG doesn't support RGBA
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Save the image in JPEG format
        img.save(output_path, 'JPEG')


def load_json(file_path):
    with open(file_path, 'r') as f:
         json_file = json.load(f)
    return json_file


def save_json(json_dict, file_path):
    with open(file_path, "w") as f:
        json.dump(json_dict, f)
    return
