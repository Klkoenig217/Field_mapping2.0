#!/usr/bin/env python 

'''
    Title: Master's thesis on AI-supported identification and localization of plants and objects
    Author: Klaudius König
    Kontact: Klaudius.koenig@web.de
    Date: 02.07.2024
    Description: Basic code of the final thesis for the recognition and localization of corn plants and artificial flowers
    Note: Only executable in conjunction with the Advanced_Navigation repository from Team FloriBot
          View on Github: https://github.com/Team-FloriBot/Advanced_Navigation/tree/ad15ba4882bd8585cbe3675614851590611205f8/src
'''

import torch
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
import tf
from sensor_msgs.msg import Image, CameraInfo
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PointStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

class ObjectDetector:

    # Initialize the setup
    def __init__(self):
        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Define global variables
        self.aligned_camera_info_msg = None
        self.aligned_image_msg = None
        self.color_image_msg = None
        self.current_time_msg = None
        self.linear_velocity_msg = 0.0
        self.angular_velocity_msg = 0.0

        # Subscribers
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.aligned_info_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.aligned_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        rospy.Subscriber("/clock", Clock, self.clock_callback)
        rospy.Subscriber("/velocity", Twist, self.velocity_callback)

        # Load YOLO model from hard disk
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/klaudius/Schreibtisch/Abschlussarbeit-Code/2D/yolov5/runs/train/exp43/weights/best.pt')

        # Define TF listener
        self.listener = tf.TransformListener()
        self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
        rospy.sleep(2)

        # Get intrinsics from depth camera info
        self.intrinsics_data = self.define_intrinsics()

        # Define variables
        self.cnt = 0
        self.previous_frame_objects = []
        self.unique_objects = []


    # Print transformation data
    def print_transform(self, source_frame, target_frame, time):
        try:
            self.listener.waitForTransform(source_frame, target_frame, time, rospy.Duration(0.5))
            (trans, rot) = self.listener.lookupTransform(source_frame, target_frame, time)
            rospy.loginfo(f"Transform from {source_frame} to {target_frame} at time {time}: translation={trans}, rotation={rot}")
            return (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
            return None, None


    # Callback functions
    def aligned_info_callback(self, msg):
        self.aligned_camera_info_msg = msg

    def aligned_callback(self, msg):
        self.aligned_image_msg = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def color_callback(self, msg):
        self.color_image_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def clock_callback(self, msg):
        self.current_time_msg = msg.clock

    def velocity_callback(self, msg):
        self.linear_velocity_msg = msg.linear.x
        self.angular_velocity_msg = msg.angular.z


    # Define RealSense intrinsics from depth camera info
    def define_intrinsics(self):
        depth_camera_info = self.aligned_camera_info_msg
        intrinsics = rs.intrinsics()
        intrinsics.width = depth_camera_info.width
        intrinsics.height = depth_camera_info.height
        intrinsics.ppx = depth_camera_info.K[2]
        intrinsics.ppy = depth_camera_info.K[5]
        intrinsics.fx = depth_camera_info.K[0]
        intrinsics.fy = depth_camera_info.K[4]
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = depth_camera_info.D
        return intrinsics


    # Object recognition function using YOLO model
    def object_recognition(self, image):
        results = self.yolo_model(image)
        boxes = results.xyxy[0][:, :4].cpu().numpy().tolist()
        class_ids = results.xyxy[0][:, -1].int().cpu().numpy()
        labels = [self.yolo_model.names[class_id] for class_id in class_ids]

        filtered_boxes = []
        filtered_labels = []
        for label, box in zip(labels, boxes):
            if label == 'stem' or label == 'flower':
                filtered_boxes.append(box)
                filtered_labels.append(label)

        return filtered_boxes, filtered_labels


    # Display the color image with detected objects
    def plot_images_with_boxes(self, image, boxes, labels, cnt, clk):
        for i, (box, label) in enumerate(zip(boxes, labels), start=1):
            xmin, ymin, xmax, ymax = map(int, box)
            color = (0, 0, 255) if label == 'flower' else (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, str(i), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            cv2.circle(image, (int(box_center[0]), int(box_center[1])), 5, (0, 0, 255), -1)

        title = f'Recognized objects in the image ({cnt}/{clk})'
        cv2.imwrite(f"/home/klaudius/Bilder_Videos/Flower/Aligned_color_{cnt}.png", image)
        cv2.waitKey(1)


    # Mark coordinates in the depth image
    def plot_depth_with_objects(self, aligned_depth_image, d2_coordinates, cnt):
        normalized_depth = cv2.normalize(aligned_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

        for obj_coordinate in d2_coordinates:
            pixel_coordinate = (int(obj_coordinate[0]), int(obj_coordinate[1]))
            cv2.circle(depth_colored, pixel_coordinate, 20, (0, 0, 255), -1)

        cv2.imwrite(f"/home/klaudius/Bilder_Videos/Flower/Depth_{cnt}.png", depth_colored)
        cv2.waitKey(1)


    # Convert 2D coordinates in the image to 3D coordinates
    def convert_coord(self, aligned_depth_image, boxes, labels):
        obj_coordinates = []
        d2_coordinates = []

        for label, box in zip(labels, boxes):
            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            aligned_depth_value = aligned_depth_image[int(box_center[1]), int(box_center[0])]
            result = rs.rs2_deproject_pixel_to_point(self.intrinsics_data, [int(box_center[0]), int(box_center[1])], aligned_depth_value)
            obj_coordinate = (result[0] / 1000.0, -result[1] / 1000.0, result[2] / 1000.0)
            d2_coordinate = (result[0] / 1000.0, -result[1] / 1000.0)
            obj_coordinates.append(obj_coordinate)
            d2_coordinates.append(d2_coordinate)

        return obj_coordinates, d2_coordinates


    # Adjust RealSense coordinates to camera_link coordinates
    def adjust_realsense_to_camera_link(self, obj_coordinate):
        x, y, z = obj_coordinate
        return (z, -x, y)


    # Convert coordinates from camera frame to base frame
    def convert_to_base_frame(self, camera_coordinates):
        base_coordinates = []
        current_time = rospy.Time.now()
        for coord in camera_coordinates:
            point_camera = PointStamped()
            point_camera.header.frame_id = 'camera_link'
            point_camera.header.stamp = current_time
            point_camera.point.x = coord[0]
            point_camera.point.y = coord[1]
            point_camera.point.z = coord[2]
            try:
                self.print_transform('camera_link', 'base_link', current_time)
                point_base = self.listener.transformPoint('base_link', point_camera)
                base_coordinates.append((point_base.point.x, point_base.point.y, point_base.point.z))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(f"Transform error1:  {e}")
        return base_coordinates


    # Convert coordinates from base frame to map frame
    def convert_to_map_frame(self, base_coordinates):
        map_coordinates = []
        current_time = rospy.Time.now()
        for coord in base_coordinates:
            point_base = PointStamped()
            point_base.header.frame_id = 'base_link'
            point_base.header.stamp = current_time
            point_base.point.x = coord[0]
            point_base.point.y = coord[1]
            point_base.point.z = coord[2]
            try:
                self.print_transform('map', 'base_link', current_time)
                point_map = self.listener.transformPoint('map', point_base)
                map_coordinates.append((point_map.point.x, point_map.point.y, point_map.point.z))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(f"Transform error2: {e}")
        return map_coordinates


    # Extract robot position and orientation
    def get_robot_position(self, trans, rot):
        x, y, _ = trans
        _, _, yaw = euler_from_quaternion(rot)
        return x, y, yaw


    # Visualize objects and robot position in RVIZ
    def visualize_objects_in_rviz(self, current_frame_objects, previous_frame_objects, unique_objects, robot_x, robot_y, robot_yaw):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)
        id_counter = 0

        # Visualize objects in the current frame - green 
        for obj in current_frame_objects:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj[0]
            marker.pose.position.y = obj[1]
            marker.pose.position.z = obj[2]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.id = id_counter
            id_counter += 1
            marker_array.markers.append(marker)

        # Visualize previous frame objects - blue
        for obj in previous_frame_objects:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj[0]
            marker.pose.position.y = obj[1]
            marker.pose.position.z = obj[2]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.id = id_counter
            id_counter += 1
            marker_array.markers.append(marker)

        # Visualize unique objects - red
        for obj in unique_objects:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj[0]
            marker.pose.position.y = obj[1]
            marker.pose.position.z = obj[2]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.id = id_counter
            id_counter += 1
            marker_array.markers.append(marker)

        # Visualize robot's position
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = robot_x
        marker.pose.position.y = robot_y
        marker.pose.position.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, robot_yaw)
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.id = id_counter
        marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.aligned_image_msg is not None and self.color_image_msg is not None:
                color_image = self.color_image_msg
                aligned_depth_image = self.aligned_image_msg

                current_time = rospy.Time.now()

                self.cnt += 1

                try:
                    #self.print_transform('base_link', 'camera_link', current_time)
                    self.listener.waitForTransform('/base_link', '/camera_link', current_time, rospy.Duration(0.5))
                    (trans_base_camera, rot_base_camera) = self.listener.lookupTransform('/base_link', '/camera_link', current_time)

                    #self.print_transform('map', 'base_link', current_time)
                    self.listener.waitForTransform('/map', '/base_link', current_time, rospy.Duration(0.5))
                    (trans_map_base, rot_map_base) = self.listener.lookupTransform('/map', '/base_link', current_time)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    print(f"Transform error: {e}")
                    continue

                # Get robot position
                robot_x, robot_y, robot_yaw = self.get_robot_position(trans_map_base, rot_map_base)

                # Recognize objects in color images
                boxes, labels = self.object_recognition(color_image)
                self.plot_images_with_boxes(color_image, boxes, labels, self.cnt, current_time)

                # Extract coordinates of objects in carmera and world coordinates
                object_coords, d2_coords = self.convert_coord(aligned_depth_image, boxes, labels)
                camera_coords = [self.adjust_realsense_to_camera_link(coord) for coord in object_coords]
                base_coords = self.convert_to_base_frame(camera_coords)
                map_coords = self.convert_to_map_frame(base_coords)
                
                # Visualize detected objects in RVIZ
                self.visualize_objects_in_rviz(map_coords, self.previous_frame_objects, self.unique_objects, robot_x, robot_y, robot_yaw)

                # Update previous and unique objects
                self.previous_frame_objects = map_coords
                self.unique_objects.extend(map_coords)

                rate.sleep()

if __name__ == "__main__":
    rospy.init_node('object_detector_node', anonymous=True)
    detector = ObjectDetector()
    detector.run()
