import os
import sys
import glob
import datetime
import threading
import argparse
import time
import numpy as np
import cv2

import rospy
import ros_numpy
import rosnode
import message_filters
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_matrix
from transforms3d.quaternions import mat2quat, quat2mat

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Fix error "ValueError: Memory growth cannot differ between GPU devices" when calling inference
# Reference: https://github.com/theAIGuysCode/yolov4-deepsort/pull/89#issue-950698392 ;
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import _init_paths

# from data import regularize_pc_point_count, depth2pc, load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps
from config_utils import load_config

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
    quat = [quat[1], quat[2], quat[3], quat[0]]
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


def set_ros_pose(pose, quat, trans):
    """
    pose is a mutable reference to a Pose() object
    quat is in (x,y,z,w) format
    Sets the fields in pose var and modifies it
    """
    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]


# class to receive pointcloud and publish grasp pose array for it
class PointToGraspPubSub:
    """
    Class for a publisher-subscriber listener to work with Contact-GraspNet

    This class follows a publisher-subscriber model where it subscribes to a
    PointCloud message consisting of object points in camera frame, processes
    them, computes feasible grasp poses for the point cloud and then publishes
    the grasps as a PoseArray message.

    The topic names for publishing and subscribing are fixed.

    Methods
    -------

    __init__()
        Provide the grasp_estimator object here. Modify other attributes here if
        needed.

    callback_points(points_pc: PointCloud message)
        callback method for ROS. Converts the ROS PointCloud message into a numpy
        array for easy processing later on and as input to the grasp network.
        Also registers (using self.step variable) whether a new point cloud msg is
        received, using which the Grasp PoseArray publisher publishes (i.e publish
        only when a new point cloud is received).

    run_network()
        Method to perform grasp sampling, adjusting for differences in gripper width
        if needed. Only samples and publishes new set of grasps if the step count
        variable is updated in callback_points(). Publishes the grasps in sorted order
        of their scores (highest scores first).
    """

    def __init__(
        self,
        tf_session,
        grasp_estimator,
        z_range,
        local_regions,
        filter_grasps,
        forward_passes,
    ):
        self.sess = tf_session
        self.estimator = grasp_estimator
        self.z_range = z_range
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps
        self.forward_passes = forward_passes

        self.xyz_image = None
        self.points_cam = None # Object PC in camera frame
        self.points_cent = None
        self.pc_all_cam = None # Entire Scene PC in camera frame
        self.depth = None
        self.frame_stamp = None
        self.frame_id = None
        self.base_frame = "base_link"
        self.SCALING_FACTOR = 1.0
        self.prev_step = None
        self.step = 0  # indicator for whether a new pc is registered
        # Create the transform the aligns Fetch with Panda Grasp
        # Apply the generated grasp pose on this transform to get pose for Fetch Gripper
        # i.e pose_fetch = pose_panda @ transform
        _quat_tf = [0.5, -0.5, 0.5, 0.5]
        _tran_tf = [0, 0, -0.1]
        self._transform_grasp = ros_qt_to_rt(_quat_tf, _tran_tf)

        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame  
        self.cv_bridge = CvBridge()

        K = [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
        intrinsics = np.array(K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print(intrinsics)
        

        # initialize a node
        rospy.init_node("pose_contact_graspnet")
        self.pose_pub = rospy.Publisher("pose_6dof", PoseArray, queue_size=10)
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        point_sub = message_filters.Subscriber("/selected_objpts", PointCloud, queue_size=5)
        pc_all_sub = message_filters.Subscriber("/all_objpts_cam", PointCloud, queue_size=5)
        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer(
            [point_sub, depth_sub], queue_size, slop_seconds
        )
        ts.registerCallback(self.callback_points)


    def callback_points(self, obj_pc_cam, depth):
        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # compute xyz image
        # height = depth_cv.shape[0]
        # width = depth_cv.shape[1]
        # xyz_image = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)

        pc_header = obj_pc_cam.header
        pc_frame_id = pc_header.frame_id
        pc_frame_stamp = pc_header.stamp
        print("[CALLBACK] Received point cloud messages!!")
        n = len(obj_pc_cam.points)
        assert n > 0
        points_cam = np.zeros((n, 3))
        for i, objpt in enumerate(obj_pc_cam.points):
            points_cam[i, :] = [objpt.x, objpt.y, objpt.z]
        print("[CALLBACK] Saved object points...")

        with lock:
            self.points_cam = points_cam.copy()
            self.frame_id = pc_frame_id
            self.frame_stamp = pc_frame_stamp
            # self.xyz_image = xyz_image.copy()
            self.depth_cv = depth_cv.copy()
            self.step += 1
        
        self.run_network(viz=False)

    def _scale(self, points, scale=1.0):
        center = np.mean(points, axis=0)
        points -= center
        points *= scale
        points += center
        return points, center


    def run_network(self, viz=False):
        # with lock:
        if listener.points_cam is None:
            return

        if self.prev_step == self.step:
            # Point cloud is not updated yet!
            return
        self.prev_step = self.step
        
        points_cam = self.points_cam.copy()

        depth_cv = self.depth_cv.copy()
        height = depth_cv.shape[0]
        width = depth_cv.shape[1]
        xyz_img = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)
       
        # xyz_img = self.xyz_image.copy()
        _depth = xyz_img[:, :, 2]
        pc_all_cam = xyz_img[(_depth > 0) & (_depth < 1.8), :]
        pc_all_cam = np.reshape(pc_all_cam, (-1, 3))
        np.save(f"pc_all_cam_{self.step}.npy", pc_all_cam)
        print(f"PC ALL: {pc_all_cam.shape}")
        size = min(60000, pc_all_cam.shape[0])
        pc_all_cam = pc_all_cam[np.random.choice(range(pc_all_cam.shape[0]), size), :]
        print(f"PC ALL: {pc_all_cam.shape}")

        frame_id = self.frame_id
        frame_stamp = self.frame_stamp
        print("\n=================================================================")
        # run the network
        print("[LISTENER] Running network...")
        print("[LISTENER] Scaling Input Points")
        # Scale the points using scaling factor before passing through the network
        center = np.mean(points_cam, axis=0)
        points_cam -= center
        points_cam *= self.SCALING_FACTOR
        points_cam += center
        print("[LISTENER] Filtering point cloud based on Z range")
        # Process the input pc as an appropriate input to the network. For our use case,
        # pc_full and pc_segments are same as we supply a single object's segmented pc.
        pc_full = pc_all_cam.copy()
        # pc_full = pc_full[
        #     (pc_full[:, 2] < self.z_range[1]) & (pc_full[:, 2] > self.z_range[0])
        # ]
        # By default, we set the keyfor pc_segment to be 0
        # The network outputs a dict with keys as the mask ids for a specific instance
        # We simply recover the recover the grasps using the dummy key we set before (INSTANCE_KEY)
        INSTANCE_KEY = 0  # some dummy key
        pc_segments = {INSTANCE_KEY: points_cam.copy()}

        print("[LISTENER] Predicting Grasps....")
        gen_grasps_d, gen_scores_d, _, _ = self.estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=self.local_regions,
            filter_grasps=self.filter_grasps,
            forward_passes=self.forward_passes,
        )
        print(f"[LISTENER] GEN GRASPS KEYS: {gen_grasps_d.keys()}")
        print(f"[LISTENER] GEN GRASPS SCORES: {gen_scores_d.keys()}")
        if -1 in gen_grasps_d.keys():
            gen_grasps, gen_scores = gen_grasps_d[-1], gen_scores_d[-1]
        else:
            gen_grasps, gen_scores = gen_grasps_d[INSTANCE_KEY], gen_scores_d[INSTANCE_KEY]
        print("[LISTENER] Sorting and Transforming Grasps....")
        # Invert the scaling for the translation part of grasp pose ; Rotation is not afffected
        gg = []
        for g in gen_grasps:
            g[:3, 3] -= center
            g[:3, 3] * (1.0 / self.SCALING_FACTOR)
            g[:3, 3] += center
            gg.append(g)
        # Arg sort the grasps using scores with highest score first
        sort_index = sorted(
            range(len(gen_scores)), key=lambda i: gen_scores[i], reverse=True
        )
        # Go along the highest grasps first and convert to Fetch using the saved transform
        sorted_graps_fetch = [gg[i] @ self._transform_grasp for i in sort_index]
        print("[LISTENER] Constructing PoseArray message for grasps....")
        # Construct a PoseArray() message
        parray = PoseArray()
        parray.header.frame_id = frame_id
        parray.header.stamp = frame_stamp  # rospy.Time.now()
        for grasp in sorted_graps_fetch:
            quat, trans = rt_to_ros_qt(grasp)
            p = Pose()
            set_ros_pose(p, quat, trans)
            parray.poses.append(p)
        # Publish the Grasps as a PoseArray ros message
        print("[LISTENER] Publishing loop....")
        while True:
            if self.pose_pub.get_num_connections() > 0:
                rospy.loginfo(
                    f"[LISTENER] Publishing Grasp Pose Array of len {len(parray.poses)}"
                )
                self.pose_pub.publish(parray)
                rospy.loginfo("[LISTENER] Finished publishing pose array")
                break
        # if viz:
        #     print("[LISTENER] Visualizing generated grasps...")
        #     visualize_grasps(pc_full, gen_grasps_d, gen_scores_d)

        print("[LISTENER] Returning from run_network() call...")
        print("=================================================================\n")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Whether to visualize the generated grasps after publishing?",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
        help="Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]",
    )
    parser.add_argument(
        "--np_path",
        default="",
        help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"',
    )
    parser.add_argument(
        "--png_path", default="", help="Input data: depth map png in meters"
    )
    parser.add_argument(
        "--K",
        default=None,
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"',
    )
    parser.add_argument(
        "--z_range",
        default=[0.1, 1.8],
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
        "--skip_border_objects",
        action="store_true",
        default=False,
        help="When extracting local_regions, ignore segments at depth map boundary.",
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=1,
        help="Run multiple parallel forward passes to mesh_utils more potential contact points.",
    )
    parser.add_argument(
        "--segmap_id",
        type=int,
        default=0,
        help="Only return grasps of the given object id",
    )
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="overwrite config parameters",
    )
    return parser


if __name__ == "__main__":
    ##--------------------------------------------------------------------------------
    parser = make_parser()
    FLAGS = parser.parse_args()
    global_config = load_config(
        FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
    )
    print(str(global_config))
    print("pid: %s" % (str(os.getpid())))
    ##--------------------------------------------------------------------------------
    ## Process the Args
    print("\n-----------------------------------------------------------------------")
    print("[INFO] Processing Args...")
    viz_grasps = FLAGS.viz
    checkpoint_dir = FLAGS.ckpt_dir
    input_paths = FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path
    z_range = eval(str(FLAGS.z_range))
    K = FLAGS.K
    local_regions = FLAGS.local_regions
    filter_grasps = FLAGS.filter_grasps
    segmap_id = FLAGS.segmap_id
    forward_passes = FLAGS.forward_passes
    skip_border_objects = FLAGS.skip_border_objects
    ##--------------------------------------------------------------------------------
    print("-----------------------------------------------------------------------")
    print("[INFO] Building Model....")
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()
    ## Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)
    ## Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    ## Load weights
    print("[INFO] Loading Weights....")
    grasp_estimator.load_weights(sess, saver, FLAGS.ckpt_dir, mode="test")
    ##--------------------------------------------------------------------------------
    ## Setup the Listener object for Publisher-Subscriber
    ## Only publishes the grasps when a new point cloud is received
    print("[INFO] Setting up the listener...")
    listener = PointToGraspPubSub(
        sess, grasp_estimator, z_range, local_regions, filter_grasps, forward_passes
    )
    print("[INFO] Starting the Subscribing-Publishing loop ...")
    # while not rospy.is_shutdown():
    #     try:
    #         listener.run_network(viz_grasps)
    #     except KeyboardInterrupt:
    #         break
    rospy.spin()
    print("[INFO] Exiting Contact GraspNet generation ROS Node")
