FROM tensorflow/tensorflow:1.12.0-gpu-py3


# Various Python and C/build deps
RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libav-tools  \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libjasper-dev \ 
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    libopencv-dev \
    python-opencv \
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

RUN pip install Cython --install-option="--no-cython-compile"
RUN pip install fasttext tflearn scikit-image scikit-learn nltk pandas gensim praw keras opencv-python

RUN pip3 install numpy

COPY jupyter_notebook_config.py /root/.jupyter/
RUN apt-get install -y python3-nose


COPY run_jupyter.sh /
RUN chmod +x /run_jupyter.sh
WORKDIR "/notebooks" 
CMD ["/run_jupyter.sh"]


