#!/usr/bin/env bash

virtualenv ./
source bin/activate
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
pip install ipython
