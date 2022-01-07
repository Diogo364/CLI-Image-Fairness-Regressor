#!/bin/bash

WORKSPACE_PATH=$( pwd )
DATA_PATH=$( pwd )/test
DOCKER_IMAGE=fairface-model-v1.0


docker run -it --rm --name fairface \
    -v $WORKSPACE_PATH:/workspace \
    -v $DATA_PATH:/data \
    $DOCKER_IMAGE \
    python /workspace/scripts/fairness_detection_cli.py \
        -i test_imgs.csv \
        -o test_outputs.csv \
        --save-clip-image /workspace/detected_faces \
        --device cpu \
        --size 300 \
        --padding 0.5