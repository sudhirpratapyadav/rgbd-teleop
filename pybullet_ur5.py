import os
import numpy as np
import pybullet as p
import pybullet_data
import math

import rospy
from geometry_msgs.msg import PoseStamped

class UR5Simulator:
    END_EFFECTOR_INDEX = 6
    RESET_JOINT_INDICES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
    RESET_JOINT_VALUES = [0.0, -1.1, 1.3, 0.0, 0.0, 0.0] + [0.45, -0.45, -0.45, 0.45]
    
    JOINT_LIMIT_LOWER = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14] + [-1.57, -1.57, -1.57, -1.57]
    JOINT_LIMIT_UPPER = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14] + [1.57, 1.57, 1.57, 1.57]
    JOINT_RANGE = [upper - lower for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER)]

    EE_POS_LIMIT = 0.15
    SIMULATION_TIME_STEP = 0.02

    def __init__(self):
        self.urdf_path = "/home/sudhir/Dropbox/git_hub_repos/computer_vision/teleop/urdf/ur5_rg2/urdf/ur5_rg2_2.urdf"
        self.physics_client = None
        self.robot_id = None
        self.movable_joints = None
        self.link_ids = None
        self.param_gripper_id = None

        rospy.init_node('OMXpybulletSim', anonymous=True)
        
        rospy.Subscriber('/command_ee_pose', PoseStamped, self.ee_cmd_pose_callback,  queue_size = 1)
        self.pose_pub = rospy.Publisher('/ee_pose', PoseStamped, queue_size=10)
        self.ee_pose_cmd = None
    
    def ee_cmd_pose_callback(self, ee_pose_cmd):
        self.ee_pose_cmd = np.array([ee_pose_cmd.pose.position.x, ee_pose_cmd.pose.position.y, ee_pose_cmd.pose.position.z,
                                     ee_pose_cmd.pose.orientation.x, ee_pose_cmd.pose.orientation.y, ee_pose_cmd.pose.orientation.z, ee_pose_cmd.pose.orientation.w])


    def deg_to_rad(self, deg):
        return [d * math.pi / 180. for d in deg]

    def rad_to_deg(self, rad):
        return [r * 180. / math.pi for r in rad]

    def quat_to_deg(self, quat):
        euler_rad = p.getEulerFromQuaternion(quat)
        return self.rad_to_deg(euler_rad)

    def deg_to_quat(self, deg):
        rad = self.deg_to_rad(deg)
        return p.getQuaternionFromEuler(rad)

    def apply_action_ik_ur5(self, target_ee_pos, target_ee_quat, target_gripper_state, num_sim_steps=5):
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.END_EFFECTOR_INDEX,
            target_ee_pos,
            solver=0,
            maxNumIterations=100,
            residualThreshold=.01
        )

        target_joint_poses = list(joint_poses[0:6])
        target_gripper_poses = [target_gripper_state, -target_gripper_state, -target_gripper_state, target_gripper_state]
        target_joint_poses.extend(target_gripper_poses)

        max_forces = [500] * len(self.movable_joints)

        p.setJointMotorControlArray(
            self.robot_id,
            self.movable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_poses,
            forces=max_forces,
            positionGains=[0.03] * len(self.movable_joints),
        )

        for _ in range(num_sim_steps):
            p.stepSimulation()

    def setup_simulation(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        p.setTimeStep(self.SIMULATION_TIME_STEP)

        print("----------------------------------------")
        print(f"Loading robot from {self.urdf_path}")
        print("----------------------------------------")
        self.robot_id = p.loadURDF(self.urdf_path)
        print("------------loaded-------------")

        num_joints = p.getNumJoints(self.robot_id)
        joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        print("------------------------------------------")
        print(f"Number of joints: {num_joints}")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_type_list[joint_info[2]]
            link_name = joint_info[12].decode("utf-8")
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]
            parent_index = joint_info[16]

            print(f"ID: {joint_id}")
            print(f"name: {joint_name}")
            print(f"type: {joint_type}")
            print(f"Link: {link_name}")
            print(f"ParentID: {parent_index}")
            print(f"lower limit: {joint_lower_limit}")
            print(f"upper limit: {joint_upper_limit}")
        print("------------------------------------------")

        self.link_ids = list(map(lambda link_info: link_info[1], p.getVisualShapeData(self.robot_id)))
        print(self.link_ids)
        link_num = len(self.link_ids)
        print(link_num)

        text_pose = list(p.getBasePositionAndOrientation(self.robot_id)[0])
        text_pose[2] += 1

        self.param_gripper_id = p.addUserDebugParameter("gripper:", -1.57, 1.57)

        self.prev_link_id = 0
        self.linkIDIn = p.addUserDebugParameter("linkID", 0, link_num-1e-3, 0)

        self.ur5_joints = [1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10, 11, 12]
        self.movable_joints = self.ur5_joints + self.gripper_joints

    def reset_robot(self):
        for i, value in zip(self.RESET_JOINT_INDICES, self.RESET_JOINT_VALUES):
            p.resetJointState(self.robot_id, i, value)

    def run_simulation(self):
        self.reset_robot()

        init_ee_position, init_ee_oreintation, _, _, _, _ = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.END_EFFECTOR_INDEX)
        ee_pos_lower = np.array(init_ee_position) - self.EE_POS_LIMIT
        ee_pos_upper = np.array(init_ee_position) + self.EE_POS_LIMIT

        while True:
            if self.ee_pose_cmd is not None:
                target_ee_pos = np.clip(self.ee_pose_cmd[:3], ee_pos_lower, ee_pos_upper)
                target_ee_quat = init_ee_oreintation
            else:
                target_ee_pos, target_ee_quat, _, _, _, _ = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.END_EFFECTOR_INDEX)

            target_gripper_state = 0.0
            self.apply_action_ik_ur5(target_ee_pos, target_ee_quat, target_gripper_state, num_sim_steps=100)
                
            ee_position, ee_orientation, _, _, _, _ = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.END_EFFECTOR_INDEX)
            # joint_positions = np.array([state[0] for state in p.getJointStates(bodyUniqueId=self.robot_id, jointIndices=self.ur5_joints)])
            self.publish_pose(ee_position, ee_orientation)
        
            linkID = p.readUserDebugParameter(self.linkIDIn)
            if linkID != self.prev_link_id:
                p.setDebugObjectColor(self.robot_id, self.link_ids[int(self.prev_link_id)], [255,255,255])
                p.setDebugObjectColor(self.robot_id, self.link_ids[int(linkID)], [255,0,0])
            self.prev_link_id = linkID

            p.stepSimulation()
    
    def publish_pose(self, ee_position, ee_orientation):
        """ Publish the end-effector pose as a PoseStamped message. """
        ee_pose = PoseStamped()
        ee_pose.header.stamp = rospy.Time.now()
        ee_pose.header.frame_id = "map"  # Adjust the frame ID as needed

        # Set the position
        ee_pose.pose.position.x = ee_position[0]
        ee_pose.pose.position.y = ee_position[1]
        ee_pose.pose.position.z = ee_position[2]

        # Orientation
        ee_pose.pose.orientation.x = ee_orientation[0]
        ee_pose.pose.orientation.y = ee_orientation[1]
        ee_pose.pose.orientation.z = ee_orientation[2]
        ee_pose.pose.orientation.w = ee_orientation[3]

        # Publish the pose
        self.pose_pub.publish(ee_pose)
        # rospy.loginfo(f"Published EE Pose: {ee_pose}")

    def cleanup(self):
        p.disconnect()

if __name__ == "__main__":
    sim = UR5Simulator()
    sim.setup_simulation()
    sim.run_simulation()
    sim.cleanup()