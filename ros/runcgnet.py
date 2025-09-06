import os
import sys
import glob
import datetime
import threading
import argparse
import time
import numpy as np
import cv2

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import _init_paths

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps
from config_utils import load_config
from transforms3d.quaternions import mat2quat, quat2mat

lock = threading.Lock()

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img

def rt_to_ros_qt(rt):
    """
    Returns (quat_xyzw, trans) from a 4x4 transform
    """
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]  # Convert to x,y,z,w
    trans = rt[:3, 3]
    return quat, trans

def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans
    return obj_T

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Whether to visualize the generated grasps?",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
        help="Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]",
    )
    parser.add_argument(
        "--obj_pc_path",
        default="./ros/obj_pc.npy",
        help="Path to .npy file containing object point cloud in meters",
    )
    parser.add_argument(
        "--scene_pc_path",
        default="./ros/scene_pc.npy",
        help="Path to .npy file containing full scene point cloud in meters",
    )
    parser.add_argument(
        "--depth_path",
        default="./ros/depth.npy",
        help="Path to .npy file containing depth map in meters",
    )
    parser.add_argument(
        "--output_path",
        default="./ros/grasps.npy",
        help="Path to save the output grasps as .npy file",
    )
    parser.add_argument(
        "--K",
        default="[527.8869068647637, 0.0, 321.7148665756361, 0.0, 524.79425074945297, 230.281919862249, 0.0, 0.0, 1.0]",
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0, 1]"',
    )
    parser.add_argument(
        "--z_range",
        default="[0.1, 4]",
        help="Z value threshold to crop the input point cloud",
    )
    parser.add_argument(
        "--local_regions",
        action="store_true",
        default=False,
        help="Crop 3D local regions around given segments.",
    )
    parser.add_argument(
        "--filter_grasps",
        action="store_true",
        default=False,
        help="Filter grasp contacts according to segmap.",
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=1,
        help="Run multiple parallel forward passes to mesh_utils more potential contact points.",
    )
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="Overwrite config parameters",
    )
    return parser

class PointToGraspProcessor:
    def __init__(
        self,
        tf_session,
        grasp_estimator,
        z_range,
        local_regions,
        filter_grasps,
        forward_passes,
        obj_pc_path,
        scene_pc_path,
        depth_path,
        output_path,
        K
    ):
        self.sess = tf_session
        self.estimator = grasp_estimator
        self.z_range = z_range
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps
        self.forward_passes = forward_passes
        self.obj_pc_path = obj_pc_path
        self.scene_pc_path = scene_pc_path
        self.depth_path = depth_path
        self.output_path = output_path
        self.SCALING_FACTOR = 1.0

        # Camera intrinsics
        K = eval(K) if isinstance(K, str) else K
        intrinsics = np.array(K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print("Camera Intrinsics:\n", self.intrinsics)

        # Transform for grasp adjustment (e.g., for Fetch gripper)
        _quat_tf = [0.5, -0.5, 0.5, 0.5]
        _tran_tf = [0, 0, -0.1]
        self._transform_grasp = ros_qt_to_rt(_quat_tf, _tran_tf)

    def _scale(self, points, scale=1.0):
        center = np.mean(points, axis=0)
        points -= center
        points *= scale
        points += center
        return points, center

    def process_grasps(self, viz=False):
        # Load point clouds and depth map from .npy files
        print("[PROCESSOR] Loading point clouds and depth map...")
        points_cam = np.load(self.obj_pc_path)  # Object point cloud
        pc_all_cam = np.load(self.scene_pc_path)  # Full scene point cloud
        print(f"[PROCESSOR] Object PC shape: {points_cam.shape}")
        print(f"[PROCESSOR] Scene PC shape: {pc_all_cam.shape}")

        np.save(f"processed_scene_pc.npy", pc_all_cam)
        print(f"[PROCESSOR] Processed scene PC shape: {pc_all_cam.shape}")
        size = min(60000, pc_all_cam.shape[0])
        pc_all_cam = pc_all_cam[np.random.choice(range(pc_all_cam.shape[0]), size), :]
        print(f"[PROCESSOR] Downsampled scene PC shape: {pc_all_cam.shape}")

        # Scale the object point cloud
        print("[PROCESSOR] Scaling input points...")
        center = np.mean(points_cam, axis=0)
        points_cam -= center
        points_cam *= self.SCALING_FACTOR
        points_cam += center

        # Prepare input for the network
        print("[PROCESSOR] Filtering point cloud based on Z range...")
        pc_full = pc_all_cam.copy()
        INSTANCE_KEY = 0  # Dummy key for single object
        pc_segments = {INSTANCE_KEY: points_cam.copy()}

        # Run the network
        print("[PROCESSOR] Predicting grasps...")
        gen_grasps_d, gen_scores_d, _, _ = self.estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=self.local_regions,
            filter_grasps=self.filter_grasps,
            forward_passes=self.forward_passes,
        )
        print(f"[PROCESSOR] Grasp keys: {gen_grasps_d.keys()}")
        print(f"[PROCESSOR] Score keys: {gen_scores_d.keys()}")

        # Extract grasps
        if -1 in gen_grasps_d.keys():
            gen_grasps, gen_scores = gen_grasps_d[-1], gen_scores_d[-1]
        else:
            gen_grasps, gen_scores = gen_grasps_d[INSTANCE_KEY], gen_scores_d[INSTANCE_KEY]

        # Adjust grasps
        print("[PROCESSOR] Sorting and transforming grasps...")
        gg = []
        for g in gen_grasps:
            g[:3, 3] -= center
            g[:3, 3] *= (1.0 / self.SCALING_FACTOR)
            g[:3, 3] += center
            gg.append(g)
        sort_index = sorted(range(len(gen_scores)), key=lambda i: gen_scores[i], reverse=True)
        sorted_grasps_fetch = [gg[i] @ self._transform_grasp for i in sort_index]

        # Convert grasps to quaternion and translation for saving
        print("[PROCESSOR] Converting grasps to quaternion and translation...")
        # grasps_array = []
        # for grasp in sorted_grasps_fetch:
        #     quat, trans = rt_to_ros_qt(grasp)
        #     grasps_array.append(np.concatenate([quat, trans]))
        grasps_array = np.array(sorted_grasps_fetch)  # Shape: [N, 7] (x,y,z,w for quat, x,y,z for trans)

        # Save grasps to .npy file
        print(f"[PROCESSOR] Saving grasps to {self.output_path}...")
        np.save(self.output_path, grasps_array)
        print(f"[PROCESSOR] Saved {len(grasps_array)} grasps.")

        # Visualize if requested
        if viz:
            print("[PROCESSOR] Visualizing generated grasps...")
            visualize_grasps(pc_full, gen_grasps_d, gen_scores_d)

        print("[PROCESSOR] Processing complete.")

if __name__ == "__main__":
    # Parse arguments
    parser = make_parser()
    FLAGS = parser.parse_args()
    global_config = load_config(
        FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
    )
    print(str(global_config))
    print("pid: %s" % (str(os.getpid())))

    # Process arguments
    print("\n-----------------------------------------------------------------------")
    print("[INFO] Processing Args...")
    viz_grasps = FLAGS.viz
    checkpoint_dir = FLAGS.ckpt_dir
    obj_pc_path = FLAGS.obj_pc_path
    scene_pc_path = FLAGS.scene_pc_path
    depth_path = FLAGS.depth_path
    output_path = FLAGS.output_path
    z_range = eval(str(FLAGS.z_range))
    K = FLAGS.K
    local_regions = FLAGS.local_regions
    filter_grasps = FLAGS.filter_grasps
    forward_passes = FLAGS.forward_passes

    # Build the model
    print("-----------------------------------------------------------------------")
    print("[INFO] Building Model...")
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Create a session and load weights
    print("[INFO] Loading Weights...")
    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    grasp_estimator.load_weights(sess, saver, FLAGS.ckpt_dir, mode="test")

    # Setup the processor
    print("[INFO] Setting up the processor...")
    processor = PointToGraspProcessor(
        sess, grasp_estimator, z_range, local_regions, filter_grasps, forward_passes,
        obj_pc_path, scene_pc_path, depth_path, output_path, K
    )

    # Process grasps
    print("[INFO] Starting grasp processing...")
    processor.process_grasps(viz_grasps)
    print("[INFO] Exiting Contact GraspNet generation.")