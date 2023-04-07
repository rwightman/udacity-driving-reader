FROM ros:melodic-perception

# CPU variant of Tensorflow
ENV TENSORFLOW_VARIANT cpu/tensorflow-1.13.1-cp27-none

# The basics
RUN apt-get update && apt-get install -q -y \
        wget \
        pkg-config \
        git-core \
	python-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Pip n Python modules
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install \
    pip==20.0.2 \
    numpy==1.16.6 \
    matplotlib==2.2.5 \
    ipykernel==4.10.1 \
    python-dateutil==2.8.1

 RUN python -m ipykernel.kernelspec

RUN pip install \
    scipy==1.2.3 \
    pandas==0.24.2 \
    jupyter==1.0.0

# Install TensorFlow
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_VARIANT}-linux_x86_64.whl
