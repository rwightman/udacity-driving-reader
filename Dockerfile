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
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py \
    && pip install python numpy matplotlib ipykernel python-dateutil --upgrade \
    && python -m ipykernel.kernelspec

RUN pip install scipy pandas jupyter

# Install TensorFlow
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_VARIANT}-linux_x86_64.whl
