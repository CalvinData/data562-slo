FROM python:3.10


# Configure the python environment (in /usr/local/bin).
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -U pip setuptools wheel \
    && pip install --no-cache-dir --upgrade -r requirements.txt


# Add and configure a non-root user to:
# - Enhance security (see: https://stackoverflow.com/questions/68155641/should-i-run-things-inside-a-docker-container-as-non-root-for-safety)
# - Avoid file-access issues when running the container locally (where container files are shared with WSL2).
# See: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user

ARG USERNAME=slo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user (drop the sudo install for production).
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Polyglot has an installation-order issue with icu, so install these manually rather than just
# listing them requirements.txt.
# See: https://stackoverflow.com/questions/64886067/polyglot-importerror-cannot-import-name-locale-from-icu
RUN pip install pyicu \
    && pip install pycld2 \
    && pip install morfessor \
    && pip install polyglot

# Add fastText.
RUN cd /code \
    && git clone https://github.com/facebookresearch/fastText.git \
    && cd fastText \
    && make

# Add some development (only) tools.
RUN echo "alias ll='ls -lahF'" >> /home/$USERNAME/.bashrc \
    && bash

USER $USERNAME
