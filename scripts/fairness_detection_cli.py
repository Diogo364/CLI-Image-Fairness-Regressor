from genericpath import isdir
import sys
from os.path import dirname, join, abspath, pardir, isdir, sep

sys.path.insert(0, abspath(join(dirname(__file__), pardir)))

import os
import dlib
import torch
import argparse
import numpy as np
import pandas as pd
from source.utils import clip_face
from source.logging import log
from source.models import FairnessDetector
from source.models import FaceDetector
from source.models import ShapePredictor

log_level = 'DEBUG'
log_output= './etc/logs/app.log'

def transform_variables(result_df, race_column='race_preds_fair', gender_column='gender_preds_fair', age_column='age_preds_fair', number_races=4):
    log(f'Transforming variables.', 'D')
    if number_races == 4:
        log(f'Using 4 features model.', 'D')
        result_df.loc[result_df[race_column] == 0, 'race'] = 'White'
        result_df.loc[result_df[race_column] == 1, 'race'] = 'Black'
        result_df.loc[result_df[race_column] == 2, 'race'] = 'Asian'
        result_df.loc[result_df[race_column] == 3, 'race'] = 'Indian'
    else:
        log(f'Using 7 features model.', 'D')
        result_df.loc[result_df[race_column] == 0, 'race'] = 'White'
        result_df.loc[result_df[race_column] == 1, 'race'] = 'Black'
        result_df.loc[result_df[race_column] == 2, 'race'] = 'Latino_Hispanic'
        result_df.loc[result_df[race_column] == 3, 'race'] = 'East Asian'
        result_df.loc[result_df[race_column] == 4, 'race'] = 'Southeast Asian'
        result_df.loc[result_df[race_column] == 5, 'race'] = 'Indian'
        result_df.loc[result_df[race_column] == 6, 'race'] = 'Middle Eastern'

    result_df.loc[result_df[gender_column] == 0, 'gender'] = 'Male'
    result_df.loc[result_df[gender_column] == 1, 'gender'] = 'Female'

    result_df.loc[result_df[age_column] == 0, 'age'] = '0-2'
    result_df.loc[result_df[age_column] == 1, 'age'] = '3-9'
    result_df.loc[result_df[age_column] == 2, 'age'] = '10-19'
    result_df.loc[result_df[age_column] == 3, 'age'] = '20-29'
    result_df.loc[result_df[age_column] == 4, 'age'] = '30-39'
    result_df.loc[result_df[age_column] == 5, 'age'] = '40-49'
    result_df.loc[result_df[age_column] == 6, 'age'] = '50-59'
    result_df.loc[result_df[age_column] == 7, 'age'] = '60-69'
    result_df.loc[result_df[age_column] == 8, 'age'] = '70+'
    
    return result_df


def output_to_df(output_dict, number_races):
    log(f'Converting dictionary to DataFrame.', 'D')
    result = pd.DataFrame(output_dict)
    return transform_variables(result, number_races=number_races)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input', required=True,
                        help='csv file of image path where col name for image path is "img_path".')
    parser.add_argument('--output', '-o', dest='output', required=True,
                        help='csv file that is going to be created with results from model.')
    parser.add_argument('--not-clip-face', dest='clip_face', action='store_false',
                        help='Flag to prevent face clipping prior to fairness detection. Recommended if using cropped faces.')
    parser.add_argument('--save-clip-image', dest='save_clip_image', type=str, default='',
                        help='Directory to save clipped images. If using --not-clip-face argument, it is not used.')
    parser.add_argument('--fairness-model', '-f', dest='fairness_model', default='assets/res34_fair_align_multi_7_20190809.pt',
                        help='path to fairness model weights.')
    parser.add_argument('--face-detector-model', dest='face_detector', default='assets/dlib_models/mmod_human_face_detector.dat',
                        help='path to face detector model weights.')
    parser.add_argument('--shape-predictor-model', dest='shape_predictor', default='assets/dlib_models/shape_predictor_5_face_landmarks.dat',
                        help='path to shape predictor landmark model weights.')
    parser.add_argument('--number-races', '-n', dest='number_races', default=7, type=int,
                        help='Number of race features outputed by the model. Should be 4 or 7.')
    parser.add_argument('--device', dest='device', type=str, default='cpu',
                        help='Process device: CPU or GPU. Should be cpu or gpu.')
    parser.add_argument('--padding', dest='padding', type=float, default=0.25,
                        help='Padding for the clipping phase. If using --not-clip-face argument, it is not used.')
    parser.add_argument('--size', dest='size', type=int, default=300,
                        help='Size for the clipping phase. If using --not-clip-face argument, it is not used.')
    args = parser.parse_args()

    log(f'CLI Variables: {vars(args)}', 'D')
    assert args.number_races in [4, 7], 'Invalid argument for --number-races, please enter 4 or 7.'
    assert args.input.endswith('csv'), 'Please, enter a valid csv file path for --input.'
    assert args.output.endswith('csv'), 'Please, enter a valid csv file path for --output.'
    assert isdir(args.save_clip_image), 'Please, enter an existing dir path for --save-clip-image.'
    assert args.device in ['cpu', 'gpu'], 'Invalid argument for --device, please enter cpu or gpu.'


    log(f'Using: {args.device}', 'D')
    device = torch.device(args.device)

    log(f'Reading input file: {args.input}', 'D')
    input_df = pd.read_csv(args.input)

    output_dict = {
        'image_names': [],
        'race_preds_fair': [],
        'gender_preds_fair': [],
        'age_preds_fair': [],
        'race_scores_fair': [],
        'gender_scores_fair': [],
        'age_scores_fair': []
    }

    log(f'Loading models', 'D')
    face_detector = FaceDetector(model_path=args.face_detector)
    shape_predictor = ShapePredictor(model_path=args.shape_predictor)
    fairness_detector = FairnessDetector(model_path=args.fairness_model)
    total_images = input_df.shape[0]
    log(f'Starting loop for images. Total:{total_images}', 'D')
    for idx, image_name in enumerate(input_df.img_path):
        log(f'Image {idx+1}/{total_images}', 'D')
        log(f'Processing image {image_name}', 'D')
        image = dlib.load_rgb_image(image_name)

        if args.clip_face:
            log(f'Entering clipping faces process', 'D')
            image = clip_face(image, face_detector, shape_predictor, args.device, size=args.size, padding=args.padding)
            if isdir(args.save_clip_image):
                dlib.save_image(image, join(args.save_clip_image, f'face_{image_name.split(sep)[-1]}'))
        if image is None:
            race_score = None
            gender_score = None
            age_score = None
            race_pred = None
            gender_pred = None
            age_pred = None
        else:
            log(f'Preprocessing image', 'D')
            image = fairness_detector.preprocess(image)
            image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image = image.to(device)

            log(f'Getting Fairness prediction', 'D')
            outputs = fairness_detector(image)
            outputs = outputs.detach().numpy()
            outputs = np.squeeze(outputs)

            race_outputs = outputs[:args.number_races]
            gender_outputs = outputs[args.number_races:args.number_races+2]
            age_outputs = outputs[args.number_races+2:args.number_races+11]

            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
            age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

            race_pred = np.argmax(race_score)
            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)

        log(f'Appending prediction to output dictionary', 'D')
        output_dict['image_names'].append(image_name)
        output_dict['race_preds_fair'].append(race_pred)
        output_dict['gender_preds_fair'].append(gender_pred)
        output_dict['age_preds_fair'].append(age_pred)
        output_dict['race_scores_fair'].append(race_score)
        output_dict['gender_scores_fair'].append(gender_score)
        output_dict['age_scores_fair'].append(age_score)

    log(f'Converting output to csv file', 'D')
    result = output_to_df(output_dict, number_races=args.number_races)    

    log(f'Saving csv file: {args.output}', 'D')
    result.to_csv(args.output, index=False)



