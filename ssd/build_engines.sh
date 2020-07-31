#!/bin/bash

set -xe

for model in ssd_mobilenet_v1_thermal ; do
    python3 build_engine.py ${model}
done
