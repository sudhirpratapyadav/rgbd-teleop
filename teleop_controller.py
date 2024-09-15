#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped

class Controller:
    def __init__(self):
        rospy.init_node('Controller')

        self.keypoint_names = rospy.get_param('human_pose/keypoints_names', {})

        self.cmd_scale = 1.0
        self.ee_pose = None
        
        # Subscribe to /human_pose_keypoints
        self.sub = rospy.Subscriber('/human_pose_keypoints', PoseArray, self.keypoints_callback)
        self.pose_pub = rospy.Publisher('/command_ee_pose', PoseStamped, queue_size=10)
        rospy.Subscriber('/ee_pose', PoseStamped, self.ee_pose_callback,  queue_size = 1)
    
    def ee_pose_callback(self, msg):
        self.ee_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                     msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    def keypoints_callback(self, msg):
        points = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses])

        rs = points[self.keypoint_names['right_shoulder']]
        rw = points[self.keypoint_names['right_wrist']]

        if not (np.isnan(rs[0]) or np.isnan(rw[0])):
            ee_cmd = rw - rs
            self.publish_pose(ee_cmd)
    
    def publish_pose(self, ee_pose_cmd):
        """ Publish the end-effector pose as a PoseStamped message. """
        ee_pose = PoseStamped()
        ee_pose.header.stamp = rospy.Time.now()
        ee_pose.header.frame_id = "map"  # Adjust the frame ID as needed

        # Set the position (convert normalized coordinates to actual scale if necessary)
        ee_pose_cmd *= self.cmd_scale
        ee_pose.pose.position.x = ee_pose_cmd[0]
        ee_pose.pose.position.y = ee_pose_cmd[1]
        ee_pose.pose.position.z = ee_pose_cmd[2]

        # Orientation is set to identity (no rotation) for simplicity
        ee_pose.pose.orientation.x = 0.0
        ee_pose.pose.orientation.y = 0.0
        ee_pose.pose.orientation.z = 0.0
        ee_pose.pose.orientation.w = 1.0

        # Publish the pose
        self.pose_pub.publish(ee_pose)
        rospy.loginfo(f"Published EE Pose: {ee_pose}")



if __name__ == '__main__':
    try:
        node = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
