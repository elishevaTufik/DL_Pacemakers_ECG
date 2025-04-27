FROM nvidia/cuda:10.1-cudnn7-devel

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

RUN apt-get update && apt-get install -y \
        libpng-dev libjpeg-dev ca-certificates \
        python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py && \
        rm get-pip.py

RUN ln -s /usr/bin/python3 /usr/bin/python

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

