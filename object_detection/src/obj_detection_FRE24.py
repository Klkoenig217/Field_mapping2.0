#!/usr/bin/env python 

'''
    Title: Master's thesis on AI-supported identification and localization of plants and objects
    Author: Klaudius König
    Kontact: Klaudius.koenig@web.de
    Date: 02.07.2024
    Description: Additional script for the tasks for the Field Robot Competition 2024
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

# Initialize OpenCV bridge outside the loop
bridge = CvBridge()


# Define global variables
aligned_camera_info_msg = None
aligned_image_msg = None
color_image_msg = None
current_time_msg = None
linear_velocity_msg = 0.0
angular_velocity_msg = 0.0


# Callback functions
def aligned_info_callback(msg):
    global aligned_camera_info_msg
    aligned_camera_info_msg = msg

def aligned_callback(msg):
    global aligned_image_msg
    aligned_image_msg = bridge.imgmsg_to_cv2(msg, "32FC1")

def color_callback(msg):
    global color_image_msg
    color_image_msg = bridge.imgmsg_to_cv2(msg, "bgr8")

def clock_callback(msg):
    global current_time_msg
    current_time_msg = msg.clock

def velocity_callback(msg):
    global linear_velocity_msg, angular_velocity_msg
    linear_velocity_msg = msg.linear.x
    angular_velocity_msg = msg.angular.z


# Define RealSense intrinsics from depth camera info
def define_intrinsics(depth_camera_info):
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
def object_recognition(image, yolo_model):
    results = yolo_model(image)
    boxes = results.xyxy[0][:, :4].cpu().numpy().tolist()
    class_ids = results.xyxy[0][:, -1].int().cpu().numpy()
    labels = [yolo_model.names[class_id] for class_id in class_ids]

    filtered_boxes = []
    filtered_labels = []
    for label, box in zip(labels, boxes):
        if label == 'stem' or label == 'flower':
            filtered_boxes.append(box)
            filtered_labels.append(label)

    return filtered_boxes, filtered_labels


# Display the color image with detected objects
def plot_images_with_boxes(image, boxes, labels, cnt, clk):
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
def plot_depth_with_objects(aligned_depth_image, boxes, labels, cnt):
    normalized_depth = cv2.normalize(aligned_depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"/home/klaudius/Bilder_Videos/Flower/depth_colored{cnt}.png", depth_colored)
    cv2.waitKey(1)


# Convert 2D coordinates in the image to 3D coordinates
def convert_coord(aligned_depth_image, intrinsics_data, boxes, labels):
    obj_coordinates = []
    for label, box in zip(labels, boxes):
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        aligned_depth_value = aligned_depth_image[int(box_center[1]), int(box_center[0])]
        result = rs.rs2_deproject_pixel_to_point(intrinsics_data, [int(box_center[0]), int(box_center[1])], aligned_depth_value)
        obj_coordinate = (result[0] / 1000.0, -result[1] / 1000.0, result[2] / 1000.0)
        obj_coordinates.append(obj_coordinate)
    return obj_coordinates


# Adjust RealSense coordinates to camera_link coordinates
def adjust_realsense_to_camera_link(obj_coordinate):
    x, y, z = obj_coordinate
    return (z, -x, y)


# Convert coordinates from camera frame to base frame
def convert_to_base_frame(camera_coordinates, listener, current_time):
    base_coordinates = []
    for coord in camera_coordinates:
        point_camera = PointStamped()
        point_camera.header.frame_id = 'camera_link'
        point_camera.header.stamp = current_time
        point_camera.point.x = coord[0]
        point_camera.point.y = coord[1]
        point_camera.point.z = coord[2]
        try:
            point_base = listener.transformPoint('base_link', point_camera)
            base_coordinates.append((point_base.point.x, point_base.point.y, point_base.point.z))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
    return base_coordinates


# Convert coordinates from base frame to map frame
def convert_to_map_frame(base_coordinates, listener, current_time):
    map_coordinates = []
    for coord in base_coordinates:
        point_base = PointStamped()
        point_base.header.frame_id = 'base_link'
        point_base.header.stamp = current_time
        point_base.point.x = coord[0]
        point_base.point.y = coord[1]
        point_base.point.z = coord[2]
        try:
            point_map = listener.transformPoint('map', point_base)
            map_coordinates.append((point_map.point.x, point_map.point.y, point_map.point.z))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
    return map_coordinates


# Extract robot position and orientation
def get_robot_position(trans, rot):
    x, y, _ = trans
    _, _, yaw = euler_from_quaternion(rot)
    return x, y, yaw


# Visualize objects and robot position in RViz
def visualize_objects_in_rviz(current_frame_objects, previous_frame_objects, unique_objects_left, unique_objects_right, marker_pub, robot_x, robot_y, robot_yaw):
    marker_array = MarkerArray()
    delete_marker = Marker()
    delete_marker.action = Marker.DELETEALL
    marker_array.markers.append(delete_marker)
    marker_pub.publish(marker_array)

    marker_array = MarkerArray()

    # Add robot position as an arrow marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 1.0
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.position.x = robot_x
    marker.pose.position.y = robot_y
    marker.pose.position.z = 0

    quaternion = tf.transformations.quaternion_from_euler(0, 0, robot_yaw)
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]
    marker.id = 1111
    marker_array.markers.append(marker)

    # Add current frame objects in green
    for i, coord in enumerate(current_frame_objects):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = coord[0]
        marker.pose.position.y = coord[1]
        marker.pose.position.z = 0
        marker.id = i + 100
        marker_array.markers.append(marker)

    # Add previous frame objects in blue
    for i, coord in enumerate(previous_frame_objects, start=len(current_frame_objects)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.position.x = coord[0]
        marker.pose.position.y = coord[1]
        marker.pose.position.z = 0
        marker.id = i + 200
        marker_array.markers.append(marker)

    # Add unique objects left row in orange
    for i, coord in enumerate(unique_objects_left, start=len(current_frame_objects) + len(previous_frame_objects)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.pose.position.x = coord[0]
        marker.pose.position.y = coord[1]
        marker.pose.position.z = 0
        marker.id = i + 300
        marker_array.markers.append(marker)

    # Add unique objects right row in red
    for i, coord in enumerate(unique_objects_right, start=len(current_frame_objects) + len(previous_frame_objects) + len(unique_objects_left)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = coord[0]
        marker.pose.position.y = coord[1]
        marker.pose.position.z = 0
        marker.id = i + 300
        marker_array.markers.append(marker)

    # Publish the markers
    marker_pub.publish(marker_array)


# Compare two sets of object coordinates
def compare_objects(current_frame_objects, previous_frame_objects, unique_objects, threshold=0.2):
    def calculate_centroid(objects):
        x_coords = [obj[0] for obj in objects]
        y_coords = [obj[1] for obj in objects]
        centroid_x = sum(x_coords) / len(objects)
        centroid_y = sum(y_coords) / len(objects)
        return (centroid_x, centroid_y, 0)

    combined_objects = current_frame_objects + previous_frame_objects
    clustered_objects = []
    processed_indices = set()

    for i, obj1 in enumerate(combined_objects):
        if i in processed_indices:
            continue
        cluster = [obj1]
        processed_indices.add(i)
        for j, obj2 in enumerate(combined_objects):
            if j in processed_indices:
                continue
            if np.linalg.norm(np.array([obj1[0], obj1[1]]) - np.array([obj2[0], obj2[1]])) < threshold:
                cluster.append(obj2)
                processed_indices.add(j)
        if len(cluster) > 5:
            clustered_objects.append(cluster)

    new_unique_objects = []

    for cluster in clustered_objects:
        centroid = calculate_centroid(cluster)
        already_unique = any(
            np.linalg.norm(np.array([centroid[0], centroid[1]]) - np.array([unique_obj[0], unique_obj[1]])) < threshold
            for unique_obj in unique_objects
        )
        if not already_unique:
            new_unique_objects.append(centroid)

    unique_objects.extend(new_unique_objects)
    
    return unique_objects, new_unique_objects


# Check the position of the detected objects and count the plants
def extend_objects(unique_objects, new_unique_objects, unique_objects_left, unique_objects_right, robot_x, robot_y, robot_yaw,row_count, current_time, previous_time):
    threshold_line_distance = 0.6 # m

    # Check if the robot has turned (row change)
    if round(robot_yaw,1) == 0.0 and (current_time - previous_time).to_sec() >= 4.0:
        print("time shift: ", (current_time - previous_time).to_sec())

        # Output the count of plants in the previous row
        print(f"Row {row_count} finished:")
        print(f"Plants right: {len(unique_objects_right)}")
        print(f"Plants left: {len(unique_objects_left)}")
            
        # Reset row objects and count
        unique_objects_left = []
        unique_objects_right = []
        row_count += 1
        previous_time = rospy.Time.now()

    # Skip data if the robot is in the middle of a turn
    if robot_yaw <= -1.9 or robot_yaw >= 1.9 or -1.7 <= robot_yaw <= 1.3:
        return unique_objects_left, unique_objects_right
    
    # Calculate the coefficients of the line equation
    a = np.tan(robot_yaw)
    b = -1
    c = robot_y - a * robot_x


    for i, obj in enumerate(new_unique_objects, start = 1):
        obj_x, obj_y, _ = obj
        
        # Calculate the perpendicular distance from the object to the line
        distance = abs(a * obj_x + b * obj_y + c) / np.sqrt(a**2 + b**2)

        print(f"obj{i} x/y: {obj_x}/{obj_y},  abs(obj_x - robot_x): {abs(obj_x - robot_x)}")

        if distance <= threshold_line_distance and 0.5 <= abs(obj_y - robot_y) <= 3.0:
            print("GET")
            # Determine if the object is left or right of the robot
            line_x = (obj_y - c) / a
            if obj_x < line_x:
                if obj not in unique_objects_left:  
                    unique_objects_left.append(obj)
            else:
                if obj not in unique_objects_right: 
                    unique_objects_right.append(obj)

    for obj in unique_objects_left:
        obj_x, obj_y, _ = obj
        print(f"unique_objects_left: {obj_x}/{obj_y}")
    for obj in unique_objects_right:
        obj_x, obj_y, _ = obj
        print(f"unique_objects_right: {obj_x}/{obj_y}")

    return unique_objects_left, unique_objects_right


# Update the list with unique objects
def add_new_objects(previous_frame_objects, obj_coordinates_in_map, robot_x, robot_y, robot_yaw):
    a = np.tan(robot_yaw)
    b = -1
    c = robot_y - a * robot_x

    def distance_to_line(x, y, a, b, c):
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    new_objects = []
    for obj in obj_coordinates_in_map:
        obj_x, obj_y, _ = obj  
        distance = distance_to_line(obj_x, obj_y, a, b, c)
        print("obj_y",obj_y)
        if distance > 0.4 and (-7.0 <= obj_y <= -3.0):
            new_objects.append(obj)
    previous_frame_objects.extend(new_objects)
    return previous_frame_objects


# Initialize ROS node and run the object detection loop
def main():
    rospy.init_node('image_listener', anonymous=True)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/klaudius/Schreibtisch/Abschlussarbeit-Code/2D/yolov5/runs/train/exp43/weights/best.pt')
                                                                       
    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, aligned_info_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, aligned_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, color_callback)
    rospy.Subscriber("/clock", Clock, clock_callback)
    rospy.Subscriber("/velocity", Twist, velocity_callback)

    listener = tf.TransformListener()
    marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
    rospy.sleep(2)

    # Get intrinsics from depth camera info
    intrinsics_data = define_intrinsics(aligned_camera_info_msg)
    
    cnt = 0
    row_count = 1
    previous_frame_objects = []
    unique_objects = []
    unique_objects_left = []
    unique_objects_right = []
    start_time =  rospy.Time.now() # Current time in s

    while not rospy.is_shutdown():
        if aligned_image_msg is None or aligned_camera_info_msg is None or color_image_msg is None or current_time_msg is None:
            continue
        
        current_time = rospy.Time.now() # Current time in s
        try:
            listener.waitForTransform('/base_link', '/camera_link', current_time, rospy.Duration(0.5))
            (trans_base_camera, rot_base_camera) = listener.lookupTransform("/base_link", "/camera_link", current_time)

            listener.waitForTransform('/map', '/base_link', current_time, rospy.Duration(0.5))
            (trans_map_base, rot_map_base) = listener.lookupTransform("map", "base_link", current_time)

            error_cnt = 0
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(f"Transform error: {e}")
            error_cnt += 1
            continue

        print("\nrospy counter: ", cnt)
        if error_cnt >= 5:
            print("Transform error counter > 5")
            break
        
        # Perform object recognition
        boxes, labels = object_recognition(color_image_msg, yolo_model)

        # Visualize objects in the color image
        plot_images_with_boxes(color_image_msg, boxes, labels, cnt, current_time)

        # Get object coordinates in RealSense frame
        obj_coordinates = convert_coord(aligned_image_msg, intrinsics_data, boxes, labels)

        # Visualize objects in the depth image
        plot_depth_with_objects(aligned_image_msg, boxes, labels, cnt)

        # Convert RealSense coordinates to camera_link frame
        obj_coordinates = [adjust_realsense_to_camera_link(coord) for coord in obj_coordinates]

        # Convert camera_link coordinates to base_link coordinates
        obj_coordinates_in_base = convert_to_base_frame(obj_coordinates, listener, current_time)

        # Convert base_link coordinates to map coordinates
        obj_coordinates_in_map = convert_to_map_frame(obj_coordinates_in_base, listener, current_time)

        # Get robot position in map frame
        robot_x, robot_y, robot_yaw = get_robot_position(trans_map_base, rot_map_base)

        # Visualize objects and robot in RViz
        visualize_objects_in_rviz(obj_coordinates_in_map, previous_frame_objects, unique_objects_left, unique_objects_right, marker_pub, robot_x, robot_y, robot_yaw)

        # Compare current and previous frame objects to find unique objects
        unique_objects, new_unique_objects = compare_objects(obj_coordinates_in_map, previous_frame_objects, unique_objects)
        unique_objects_left, unique_objects_right = extend_objects(unique_objects, new_unique_objects, unique_objects_left, unique_objects_right, robot_x, robot_y, robot_yaw, row_count, current_time, start_time)

        # Update previous frame objects
        previous_frame_objects = add_new_objects(previous_frame_objects, obj_coordinates_in_map, robot_x, robot_y, robot_yaw)
        #previous_frame_objects.extend(map_coord)

        cnt += 1
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
        print("log error")
    except rospy.ROSInterruptException:
        pass