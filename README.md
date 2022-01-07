# Image Fairness Regressor

## Description
This CLI program is an implementation of the [FairFace DeepLearning model](#References) that provides sensitive annotation - race, gender and age - based on images.

## Why should one use this
### Fairness evaluation over Model
If one is in need of evaluating fairness over a ML model based on images of people, one might need to use an annotated dataset. In order to generate fast labeled data from unlabeled images, this application is very handy.

### Fairness evaluation over Dataset
If one is in need of assess the level of diversity of an unlabeled image dataset.

## Table of contents
- [Image Fairness Regressor](#image-fairness-regressor)
  - [Description](#description)
  - [Why should one use this](#why-should-one-use-this)
    - [Fairness evaluation over Model](#fairness-evaluation-over-model)
    - [Fairness evaluation over Dataset](#fairness-evaluation-over-dataset)
  - [Table of contents](#table-of-contents)
  - [How to use](#how-to-use)
    - [1. Link to FairFace pre-trained model [Must Download]:](#1-link-to-fairface-pre-trained-model-must-download)
    - [2. CLI Application](#2-cli-application)
      - [Description:](#description-1)
      - [Default pipeline:](#default-pipeline)
      - [CLI Parameters:](#cli-parameters)
        - [Mandatory:](#mandatory)
        - [Optional:](#optional)
      - [Example of usage:](#example-of-usage)
      - [Instalation](#instalation)
    - [Docker](#docker)
      - [1. Building the Image:](#1-building-the-image)
      - [2. Running the Container:](#2-running-the-container)
        - [Pointers:](#pointers)
  - [Technologies](#technologies)
  - [Logs](#logs)
  - [References](#references)
  - [Autor](#autor)

---
## How to use
### 1. Link to FairFace pre-trained model [Must Download]:
- The pre-trained model is available in the README File from https://github.com/dchen236/FairFace.

### 2. CLI Application
- Appliation File: `scripts/fairness_detection_cli.py`

#### Description:
The CLI application uses an `--input` CSV file containing the the full path to all images to be processed under the column name of `"img_path"` to load each image and run the FairFace Model over it. All outputs are saved at the end under another CSV file specified in the `--output` parameter.

#### Default pipeline:
1. Loading input image;
2. Detect the face using the Face Detector;
3. Generate face landmarks using the Shape Predictor Landmark;
4. Crop image using a padding over the predicted landmarks;
5. Fairness Regressor;

#### CLI Parameters:
##### Mandatory:
- `--input`, `-i`: 
  - CSV file containing the path to all images within the column name `"img_path"`.
- `--output`, `-o`:
    - Valid path to the output CSV file containing the model's results.


##### Optional:
- `--fairness-model`, `-f`:
  - Path to the fairness model weights.
  - Default: `'assets/res34_fair_align_multi_7_20190809.pt'`
- `--number-races`, `-n`:
  - Number of race features outputed by the model. Should be 4 or 7.
  - Default=`7`
  > **IMPORTANT:** Use the same number of race classes from the loaded model version in the `--fairness-model` Parameter. 
- `--device`:    
  - Process device: CPU or GPU. Should be cpu or gpu.
  - Default: `'cpu'`
- `--not-clip-face`:
  - Flag to prevent face clipping prior to fairness detection. 
  > Tip: Recommended if already using cropped faces.
- `--padding`:
  - Padding used for the clipping phase.
  - Default: `0.25`
  > Warning: If using --not-clip-face argument, it is not used.
- `--size`:           
  - Size resize clipped image.
  - Default: `300`
  > Warning: If using --not-clip-face argument, it is not used.
- `--face-detector-model`:
  - Path to the Face detector model weights.
  - Default: `'assets/dlib_models/mmod_human_face_detector.dat'`
  > Warning: If using --not-clip-face argument, it is not used.
- `--shape-predictor-model`:
  - Path to shape predictor landmark model weights.
  - Default: `'assets/dlib_models/shape_predictor_5_face_landmarks.dat'`
  > Warning: If using --not-clip-face argument, it is not used.
- `--save-clip-image`:
  - Directory to save clipped the images. 
  > *If using --not-clip-face argument, it is not used.*

#### Example of usage:
> ```$ python scripts/fairness_detection_cli.py --input ./input.csv --output output.csv -n 7 --device cpu --padding 0.30 --save-clip-image ./clipped-output-folder/ -f ./models/res34_fair_align_multi_7_20190809.pt```

---
#### Instalation
All requirements are in the `requirements.txt` file. From pip, just run in the terminal the following command and you are ready to go.
> ```$ pip install -r requirements.txt```

### Docker
The shellscript files under the project root were made to simplify the process of using the project with Docker, therefore one only need to follow the steps below:

#### 1. Building the Image:
Simply run the `docker_build.sh` script. It will automatically map your username, user id and group id to the image.
```$ ./docker_build.sh```

#### 2. Running the Container:
Within the `docker_run.sh` script are all the parameters and volumes needed in order to run the application from a docker container. Just need to follow the pointers below:

```$ ./docker_run.sh```

##### Pointers:
**1. Make sure to mount all volumes needed correctly.**
If the images needed are in the `/home/user/dataset` directory, the container can only access it if the volume is mounted.

**2. All paths from the input CSV file must follows the CONTAINER.**
If the script is running inside a container, it needs to find the images within it. Make sure to map correctly the path all images are going to be.



## Technologies

- `Python`
- `OpenCV`
- `Docker`
- `Dlib`
- `PyTorch`

## Logs
All logs can be accessed from the `./etc/logs/app.log` file within the project root.

## References

**FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age**

Paper: https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf

Github: https://github.com/dchen236/FairFace

Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).

```
 @inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
```



## Autor

<a href="https://github.com/Diogo364" >
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/44041957?s=400&u=44d208aa5d0b6df75c0bb60e2583fe6015cc0ed0&v=4" width="100px;" alt=""/>
</a>
<br>

[Diogo Nascimento](https://github.com/Diogo364)
[![Linkedin Badge](https://img.shields.io/badge/-Diogo-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/diogo-telheiro-do-nascimento/)](https://www.linkedin.com/in/diogo-telheiro-do-nascimento/) 
[![Gmail Badge](https://img.shields.io/badge/-diogotnascimento94@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:diogotnascimento94@gmail.com)](mailto:diogotnascimento94@gmail.com)