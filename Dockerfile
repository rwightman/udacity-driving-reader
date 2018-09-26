FROM ros:kinetic-perception

# CPU Python 3.5 variant of Tensorflow
ENV TENSORFLOW_VARIANT cpu/tensorflow-0.11.0rc0-cp27-none

# The basics 
RUN apt-get update && apt-get install -q -y \
        wget \
        pkg-config \
        git-core \
        python \
        python-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*    

# Install Pip n Python modules
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py \
    && \ 
    pip --no-cache-dir install \
        scipy \
        numpy \
        matplotlib \
        pandas \
        ipykernel \
        jupyter \
        pyyaml \
    && \
    pip install python-dateutil --upgrade \
    && \
    python -m ipykernel.kernelspec

# Install TensorFlow
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_VARIANT}-linux_x86_64.whl
