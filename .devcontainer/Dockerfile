# Base Image
FROM nvcr.io/nvidia/jax:25.01-py3


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY /.devcontainer/requirements.txt /app


# https://pytensor.readthedocs.io/en/latest/troubleshooting.html#how-do-i-configure-test-my-blas-library
# https://superfastpython.com/numpy-number-blas-threads/
# export PYTENSOR_FLAGS="blas__ldflags='-llapack -lblas -lcblas'"
# export OMP_NUM_THREADS=
ENV PYTENSOR_FLAGS="blas__ldflags='-llapack -lblas -lcblas'"


# Environment
SHELL [ "/bin/bash", "-c" ]


# Installation
RUN apt update && apt -q -y upgrade && apt -y install sudo && sudo apt -y install graphviz && \
    sudo apt -y install wget && sudo apt -y install curl && sudo apt -y install unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp/ && cd /tmp && sudo ./aws/install && cd ~ && \
    pip install --upgrade pip && \
    pip install --requirement /app/requirements.txt --no-cache-dir && \
    echo $PYTENSOR_FLAGS


# Port
EXPOSE 8000 8888


# ENTRYPOINT
ENTRYPOINT ["bash"]
