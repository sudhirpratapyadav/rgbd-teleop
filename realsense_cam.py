import pyrealsense2 as rs
import numpy as np
import cv2
import open3d

class RealSenseCam:
    def __init__(self, width=640, height=480, fps=30, depth_enabled=True):
        """ Initialize the RealSense camera manager. """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.frame_count = 0
        self.frame_time = 0
        self.frame_rate = 0

        self.depth_enabled = depth_enabled

        self.allowed_resolutions = [(1280, 800),
                                    (1280, 720),
                                    (640, 480),
                                    (640, 360),
                                    (480, 270),
                                    (424, 240)]

    def _create_pipeline(self):
        """ Create RealSense pipeline """

        if (self.width, self.height) not in self.allowed_resolutions:
            raise ValueError(f"Invalid resolution: {self.width}:{self.height}\nAllowed resolutions: {self.allowed_resolutions}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        if self.depth_enabled:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)


    def start(self):
        """ Open the camera. """
        if not self.is_device_available():
            raise ValueError("No RealSense device connected.")

        self._create_pipeline()
        # Start streaming
        profile = self.pipeline.start(self.config)

        if self.depth_enabled:

            # Setup the 'High Accuracy'-mode
            depth_sensor = profile.get_device().first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
                print('%02d: %s'%(i,visulpreset))
                if visulpreset == "High Accuracy":
                    depth_sensor.set_option(rs.option.visual_preset, i)

            # enable higher laser-power for better detection
            depth_sensor.set_option(rs.option.laser_power, 180)

            # lower the depth unit for better accuracy and shorter distance covered
            depth_sensor.set_option(rs.option.depth_units, 0.005)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            self.depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , self.depth_scale)

            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.cam_mtx = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],[0.0, intrinsics.fy, intrinsics.ppy],[0.0, 0.0, 1.0]])
            self.cam_dist = np.array(intrinsics.coeffs).reshape(1,5)
            self.open3d_camera_info = open3d.camera.PinholeCameraIntrinsic(self.width, self.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            self.align = rs.align(rs.stream.color)

    def stop(self):
        """ Close the camera. """
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None

    def read_frame(self):
        """ Read a frame from the camera. """
        frames = self.pipeline.wait_for_frames()

        if self.depth_enabled:
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return False, None, None
            
            self.frame_count += 1
            self.frame_time = cv2.getTickCount()
            
            color_image = np.asanyarray(color_frame.get_data())

            return True, color_image, depth_frame

        else:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            
            self.frame_count += 1
            self.frame_time = cv2.getTickCount()
            
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image
    
    def get_depth_colormap(self, depth_frame):
        return cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
    
    def get_point_cloud(self, color_img, depth_frame):

        depth_img = np.asanyarray(depth_frame.get_data())

        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color=open3d.geometry.Image(color_img),
            depth=open3d.geometry.Image(depth_img),
            depth_scale=1.0/self.depth_scale,
            convert_rgb_to_intensity=False)
        
        pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=self.open3d_camera_info,
            )
        # pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        # print(open3d_point_cloud)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return pcd
        
    def is_opened(self):
        """ Check if the camera is open. """
        return self.pipeline is not None

    def is_device_available(self):
        # """ Check if a RealSense device is available. """
        devices = rs.context().query_devices()
        if len(devices) == 0:
            return False
        else:
            try:
                device_name = devices.front().get_info(rs.camera_info.name)
                print("RealSense device found:", device_name)
                return True
            except:
                return False
    

# Stand alone test
if __name__ == "__main__":

    import argparse
    import struct
    import rospy
    from sensor_msgs.msg import PointCloud2, PointField
    # from sensor_msgs import point_cloud2
    import sensor_msgs.point_cloud2 as pc2
    from std_msgs.msg import Header

    from ctypes import * # convert float to uint32

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

    # Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
    def convert_point_cloud_to_ros(open3d_cloud, frame_id="map"):
        # Set "header"
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Set "fields" and "cloud_data"
        points=np.asarray(open3d_cloud.points)
        if not open3d_cloud.colors: # XYZ only
            fields=FIELDS_XYZ
            cloud_data=points
        else: # XYZ + RGB
            fields=FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(open3d_cloud.colors)*255)
            colors = colors.astype(np.uint32)
            colors = colors[:,2] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,0]  
            colors = colors.view(np.float32)
            cloud_data = [tuple((*p, c)) for p, c in zip(points, colors)]
        # create ros_cloud
        return pc2.create_cloud(header, fields, cloud_data)

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', action='store_true', help='Enable depth')
    parser.add_argument('--pointcloud', action='store_true', help='Enable point cloud')
    args = parser.parse_args()
    if args.pointcloud:
        args.depth = True
    cam = RealSenseCam(depth_enabled=args.depth)
    cam.start()

    rospy.init_node('realsense_publisher', anonymous=True)
    pointcloud_pub = rospy.Publisher('/cam_pointcloud', PointCloud2, queue_size=10)


    # if args.pointcloud:
    #     vis = open3d.visualization.Visualizer()
    #     vis.create_window(window_name="Open3D Window", width=640, height=480)
    #     geometry = open3d.geometry.PointCloud()
    #     render_option = vis.get_render_option()
    #     render_option.point_size = 1.0
    #     geom_added = False

    while cam.is_opened():
        if cam.depth_enabled:
            success, color_image, depth_frame = cam.read_frame()
        else:
            success, color_image = cam.read_frame()
        if success:
            if cam.depth_enabled:
                depth_colomap = cam.get_depth_colormap(depth_frame)
                display_frame = np.hstack((color_image, depth_colomap))
                if args.pointcloud:
                    point_cloud = cam.get_point_cloud(color_image, depth_frame)

                    ros_pointcloud = convert_point_cloud_to_ros(point_cloud)
                    pointcloud_pub.publish(ros_pointcloud)

                    # geometry.points = point_cloud.points
                    # geometry.colors = point_cloud.colors
                    # if not geom_added:
                    #     vis.add_geometry(geometry)
                    #     geom_added = False
                    # else:
                    #     vis.update_geometry(geometry)
                    # vis.poll_events()
                    # vis.update_renderer()
            else:
                display_frame = color_image
            cv2.imshow("frame", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.stop()
    cv2.destroyAllWindows()
    # if args.pointcloud:
    #     vis.destroy_window()
