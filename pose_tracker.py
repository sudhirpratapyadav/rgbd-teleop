#!/usr/bin/env python

import argparse
import math
import numpy as np
import rospy
import cv2
import mediapipe as mp
from mediapipe import solutions
import time
from web_cam import WebCam
from realsense_cam import RealSenseCam
from geometry_msgs.msg import Pose, PoseArray

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from ctypes import * # convert float to uint32

from timing import TimingLogger
from scipy.spatial.transform import Rotation as R

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

def is_valid(value: float) -> bool:
    return 0 <= value <= 1 or math.isclose(value, 0) or math.isclose(value, 1)

class PoseTrackerROSNode:
    def __init__(self, cam_type='realsense', depth_enabled=False, model_name='lite'):
        """ Initialize the HandTracker with the specified camera type and ROS components. """
        
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=f'/home/sudhir/Dropbox/git_hub_repos/computer_vision/teleop/pose_landmarker_{model_name}.task'),
            running_mode=mp.tasks.vision.RunningMode.VIDEO)
        
        self.cam = self._initialize_camera(cam_type, depth_enabled)
        preview_scale = 1
        self.preview_resolution = (math.floor(self.cam.width*preview_scale), math.floor(self.cam.height*preview_scale))

        self.pub_keypoints = rospy.Publisher('/human_pose_keypoints', PoseArray, queue_size=10)

        self._set_ros_params()
        self.depth_range = (0.2, 3.0)

        self.pcd_pub = rospy.Publisher('/cam_pointcloud', PointCloud2, queue_size=1)

        # self.tranform_pcd = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ])
        self.tranform_pcd = np.dot(R.from_euler('x', 90, degrees=True).as_matrix(), R.from_euler('z', 180, degrees=True).as_matrix())

        rospy.loginfo("HandTracker node initialized.")

    def _initialize_camera(self, cam_type, depth_enabled):
        """ Initialize the camera based on the specified type. """
        if cam_type == 'realsense':
            cam = RealSenseCam(depth_enabled=depth_enabled)
            self.depth_enabled = cam.depth_enabled
            if not cam.is_device_available():
                rospy.logwarn("RealSense camera not found. Using WebCam instead.")
                return WebCam()
            return cam
        else:
            self.depth_enabled = False
            return WebCam()
    
    def _set_ros_params(self):
        keypoints_names = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
            'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
            'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
            'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
            'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
            'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
            'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
            'right_foot_index': 32
        }
        connections = list(solutions.pose.POSE_CONNECTIONS)
        # keypoints_str = {str(k): v for k, v in keypoints.items()}
        rospy.set_param('human_pose/keypoints_names', keypoints_names)
        rospy.set_param('human_pose/connections', connections)
    
    def draw_landmarks_on_image(self, color_img, depth_frame, keypoints_2d):
        circle_radius = 2
        circle_border_radius = max(circle_radius + 1, int(circle_radius * 1.2))
        thickness = 1
        if depth_frame is not None:
            depth_image = self.cam.get_depth_colormap(depth_frame)
        for connection in solutions.pose.POSE_CONNECTIONS:
            start = keypoints_2d[connection[0]]
            end = keypoints_2d[connection[1]]
            if not (start is None or end is None):
                cv2.line(color_img, start, end, (255, 255, 255), thickness)
                if depth_frame is not None:
                    cv2.line(depth_image, start, end, (255, 255, 255), thickness)
                    
        for point in keypoints_2d:
            if point is not None:
                cv2.circle(color_img, point, circle_border_radius, (255, 255, 255), thickness)
                cv2.circle(color_img, point, circle_radius, (0, 255, 0), thickness)
                if depth_frame is not None: 
                    cv2.circle(depth_image, point, circle_border_radius, (255, 255, 255), thickness)
                    cv2.circle(depth_image, point, circle_radius, (0, 0, 255), thickness)
                    
        
        color_img = cv2.flip(cv2.resize(color_img, self.preview_resolution), 1)
        if depth_frame is not None:
            depth_image = cv2.flip(cv2.resize(depth_image, self.preview_resolution), 1)
            disp_img = np.hstack((color_img, depth_image))
        else:
            disp_img = color_img
        return disp_img
        
    
    def calc_key_points(self, pos_res, depth_frame=None):
        keypoints_3d = []
        keypoints_2d = []
        if depth_frame is None:
            for landmark_2d, landmark_3d in zip(pos_res.pose_landmarks[0], pos_res.pose_world_landmarks[0]):
                if landmark_2d.visibility > _VISIBILITY_THRESHOLD and landmark_2d.presence > _PRESENCE_THRESHOLD and is_valid(landmark_2d.x) and is_valid(landmark_2d.y):
                    px = min(math.floor(landmark_2d.x * self.cam.width), self.cam.width - 1)
                    py = min(math.floor(landmark_2d.y * self.cam.height), self.cam.height - 1)
                    keypoints_2d.append([px, py])
                    keypoints_3d.append([landmark_3d.x, landmark_3d.y, landmark_3d.z])
                else:
                    keypoints_3d.append([np.nan, np.nan, np.nan])
                    keypoints_2d.append(None)
        else:
            for landmark in pos_res.pose_landmarks[0]:
                if landmark.visibility > _VISIBILITY_THRESHOLD and landmark.presence > _PRESENCE_THRESHOLD and is_valid(landmark.x) and is_valid(landmark.y):
                    px = min(math.floor(landmark.x * self.cam.width), self.cam.width - 1)
                    py = min(math.floor(landmark.y * self.cam.height), self.cam.height - 1)
                    keypoints_2d.append([px, py])

                    kp_depth = depth_frame.get_distance(px, py)
                    if kp_depth < self.depth_range[0] or kp_depth > self.depth_range[1]:
                        keypoints_3d.append([np.nan, np.nan, np.nan])
                        continue
                    xyz_img = kp_depth*np.array([px,py,1])
                    xyz_c = np.matmul(np.linalg.inv(self.cam.cam_mtx), xyz_img.T)
                    keypoints_3d.append(xyz_c.reshape(3).tolist())
                else:
                    keypoints_3d.append([np.nan, np.nan, np.nan])
                    keypoints_2d.append(None)
        keypoints_3d = np.array(keypoints_3d)
        if depth_frame is None:
            keypoints_3d = keypoints_3d.dot(self.tranform_pcd)
        return keypoints_2d, keypoints_3d
    
    def publish_keypoints(self, keypoints_3d):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'map'
        for point in keypoints_3d:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            pose_array.poses.append(pose)
        self.pub_keypoints.publish(pose_array)
    
    def publish_point_cloud(self, color_img, depth_frame, frame_id='map'):

        o3d_pcd = self.cam.get_point_cloud(color_img, depth_frame)
        
        # Set "header"
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Set "fields" and "cloud_data"
        points=np.asarray(o3d_pcd.points)
        if not o3d_pcd.colors: # XYZ only
            fields=FIELDS_XYZ
            cloud_data=points
        else: # XYZ + RGB
            fields=FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(o3d_pcd.colors)*255)
            colors = colors.astype(np.uint32)
            colors = colors[:,2] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,0]  
            colors = colors.view(np.float32)
            cloud_data = [tuple((*p, c)) for p, c in zip(points, colors)]
        
        self.pcd_pub.publish(pc2.create_cloud(header, fields, cloud_data))
        
    def start_tracking(self):
        """ Start the hand tracking process using the selected camera. """
        self.cam.start()
        prev_time = time.time()
        time_log = TimingLogger()
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(self.options) as pose_model:
                while not rospy.is_shutdown() and self.cam.is_opened():
                    time_log.next()
                    if self.depth_enabled:
                        success, color_image, depth_frame = self.cam.read_frame()
                        # self.publish_point_cloud(color_image, depth_frame)
                    else:
                        success, color_image = self.cam.read_frame()
                        depth_frame = None
                    if not success:
                        continue
                    time_log.stamp('read_frame')
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
                    pos_res = pose_model.detect_for_video(mp_image, int(time.time() * 1000))
                    time_log.stamp('pose_detect')

                    if pos_res.pose_landmarks:
                        key_points_2d, key_points_3d = self.calc_key_points(pos_res, depth_frame)
                        disp_img = self.draw_landmarks_on_image(color_image, depth_frame, key_points_2d)
                        self.publish_keypoints(key_points_3d)
                    else:
                        disp_img = cv2.flip(cv2.resize(color_image, self.preview_resolution), 1)
                    time_log.stamp('keypoints')

                    # Calculate FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_fps = fps
                    prev_time = curr_time
                    fps_text = f'FPS: {int(fps)}'
                    cv2.putText(disp_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('MediaPipe Holistic', disp_img)

                    time_log.stamp('display')
                    
                    # Exit on 'ESC' key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            time_log.report(ignore_init_iters=3, include_iters=False)
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()

def main():
    # take argmuent for depth
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', action='store_true', help='Enable depth')
    parser.add_argument('--model', type=str, default='lite', help='Model name {lite, full, heavy}')
    args = parser.parse_args()

    rospy.init_node('pose_tracker_node', anonymous=True)
    cam_type = rospy.get_param('~cam', 'realsense')  # Use ROS parameter for camera type
    hand_tracker = PoseTrackerROSNode(cam_type=cam_type, depth_enabled=args.depth, model_name=args.model)
    hand_tracker.start_tracking()

if __name__ == "__main__":
    main()
