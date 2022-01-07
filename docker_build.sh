#!/bin/bash

docker build \
-t fairface-model-v1.0 \
--build-arg USERNAME=$(whoami) \
--build-arg USER_UID=$(id -u) \
--build-arg USER_GID=$(id -g) \
.