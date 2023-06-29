#!/usr/bin/env bash

python -m venv .venv
source .venv/bin/activate
pip install opencv-python pycocotools matplotlib onnxruntime onnx jupyterlab