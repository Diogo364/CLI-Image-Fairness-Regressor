FROM pytorch/pytorch:latest

LABEL Name=fairface Version=1.0

ARG USERNAME=user
ARG USER_UID=1001
ARG USER_GID=1001

ENV TORCH_HOME=/workspace/etc/.cache

RUN apt-get -y update && apt-get install -y \
        ffmpeg libsm6 libxext6     \
        wget bzip2 ca-certificates \
        cmake build-essential      \
        pkg-config                 \
        python-dev                \
        python-pip                 \
        python-setuptools          \
        python-virtualenv          \
        && apt-get clean

RUN mkdir /etc/sudoers.d/
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USERNAME} -m ${USERNAME}
RUN echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN chmod -R a+rwx /workspace
WORKDIR /workspace

RUN mkdir /data && chmod -R a+rwx /data
RUN chmod -R a+rwx /etc

RUN pip install --upgrade pip
RUN pip --no-cache-dir install dlib\
                                pandas

USER ${USERNAME}

