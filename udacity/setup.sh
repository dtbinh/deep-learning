#!/usr/bin/env bash

pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
pip install ipython
pip install jupyter
pip install jupyter-themer

jupyter-themer -c neo -l wide

