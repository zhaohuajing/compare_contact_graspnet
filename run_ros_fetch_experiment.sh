#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python ./ros/test_model_ros.py --local_regions --filter_grasps --forward_passes 4 --viz
