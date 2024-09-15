#!/usr/bin/env python
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped

class HumanPoseVisualizer:
    def __init__(self):
        rospy.init_node('human_pose_visualizer')

        # Load parameters (keypoint names and connections)
        self.keypoint_names = rospy.get_param('human_pose/keypoints_name', {})
        self.connections = rospy.get_param('human_pose/connections', [])
        
        # Subscribe to /human_pose_keypoints
        rospy.Subscriber('/human_pose_keypoints', PoseArray, self.keypoints_callback)
        rospy.Subscriber('/command_ee_pose', PoseStamped, self.ee_cmd_pose_callback,  queue_size = 1)
        rospy.Subscriber('/ee_pose', PoseStamped, self.ee_pose_callback,  queue_size = 1)

        self.ee_pose = None
        self.ee_cmd_pose = None

        # Publish to /human_pose_markers for RViz visualization
        self.pub = rospy.Publisher('/human_pose_markers', MarkerArray, queue_size=10)
    
    def ee_pose_callback(self, msg):
        self.ee_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                     msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    
    def ee_cmd_pose_callback(self, msg):
        self.ee_cmd_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                     msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    
    def create_point_marker(self, id, point, action):
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust this to your camera's frame
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = action
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker

    def create_line_marker(self, id, start, end, action):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.type = Marker.LINE_STRIP
        marker.action = action
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Line width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.points.append(Point(start[0], start[1], start[2]))
        marker.points.append(Point(end[0], end[1], end[2]))
        return marker

    def keypoints_callback(self, msg):
        marker_array = MarkerArray()
        points = [(pose.position.x, pose.position.y, pose.position.z) for pose in msg.poses]

        # Create point markers
        m_id = 0
        for point in points:
            if np.isnan(point[0]):
                action = Marker.DELETE
            else:
                action = Marker.ADD
            marker = self.create_point_marker(m_id, point, action)
            marker_array.markers.append(marker)
            m_id += 1
        
        # Create line markers for connections
        for connection in self.connections:
            start = points[connection[0]]
            end = points[connection[1]]
            if np.isnan(start[0]) or np.isnan(end[0]):
                action = Marker.DELETE
            else:
                action = Marker.ADD
            marker = self.create_line_marker(m_id, start, end, action)
            marker_array.markers.append(marker)
            m_id += 1

        # Create markers for end-effector pose and command
        if self.ee_pose is not None:
            marker = self.create_point_marker(m_id, self.ee_pose[:3], Marker.ADD)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker_array.markers.append(marker)
            m_id += 1
        
        if self.ee_cmd_pose is not None:
            marker = self.create_point_marker(m_id, self.ee_cmd_pose[:3], Marker.ADD)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
            m_id += 1
        
        # line
        if self.ee_pose is not None and self.ee_cmd_pose is not None:
            marker = self.create_line_marker(m_id, self.ee_pose[:3], self.ee_cmd_pose[:3], Marker.ADD)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
            m_id += 1

        # Publish the marker array to RViz
        self.pub.publish(marker_array)

if __name__ == '__main__':
    try:
        node = HumanPoseVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
