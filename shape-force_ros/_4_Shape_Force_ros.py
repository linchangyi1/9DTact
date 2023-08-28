import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import cv2
import yaml
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import WrenchStamped
import numpy as np
from shape_reconstruction import Sensor
from sf_visualizer import Visualizer

initialized_wrench = np.array([.5, .5, 1.0, .5, .5, .5])
wrench = initialized_wrench


def wrench_callback(msg):
    global wrench
    wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                       msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])


if __name__ == '__main__':
    rospy.init_node('9DTact')
    wrench_sub = rospy.Subscriber('/predicted_wrench', WrenchStamped, wrench_callback)
    representation_pub = rospy.Publisher('/deformation_representation', Image, queue_size=1)
    rate = rospy.Rate(90)

    f = open("../shape_reconstruction/shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor = Sensor(cfg)
    visualizer = Visualizer(sensor.points)

    while sensor.cap.isOpened() and (not rospy.is_shutdown()):
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('RawImage_GRAY', img_GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        depth_map = sensor.height_map_2_depth_map(height_map)
        cv2.imshow('DepthMap', depth_map)
        height_map = sensor.expand_image(height_map)

        representation, mixed_visualization = sensor.raw_image_2_representation(img_GRAY)
        cv2.imshow('Representation', mixed_visualization)

        if height_map.max() > 0.1:
            img_msg = CvBridge().cv2_to_imgmsg(representation)
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = 'representation'
            representation_pub.publish(img_msg)
        else:
            wrench = initialized_wrench
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if not visualizer.vis.poll_events():
            break
        else:
            points, gradients = sensor.height_map_2_point_cloud_gradients(
                height_map)
            visualizer.update(points, gradients, np.copy(wrench))
        rate.sleep()
