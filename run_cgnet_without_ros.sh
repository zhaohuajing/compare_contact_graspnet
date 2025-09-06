#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"


python ./ros/runcgnet.py --local_regions --filter_grasps --forward_passes 4 --viz --obj_pc_path "/home/robot-nav/obj_pc.npy"  --scene_pc_path "/home/robot-nav/scene_pc.npy"  --output_path "/home/robot-nav/grasps.npy"
# python ./ros/runcgnet.py --local_regions --filter_grasps --forward_passes 4 --viz 