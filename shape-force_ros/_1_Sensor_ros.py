import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import cv2
import rospy
from cv_bridge import CvBridge
import yaml
from sensor_msgs.msg import Image
from shape_reconstruction import Sensor


if __name__ == '__main__':
    f = open("ros_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor = Sensor(cfg)

    rospy.init_node('sensor', anonymous=True)

    # the latch is set to be True for the latter subscribers
    rectify_crop_ref_img_publisher = rospy.Publisher('/rectify_crop_ref_image', Image, queue_size=1, latch=True)
    rectify_crop_image_publisher = rospy.Publisher('/rectify_crop_image', Image, queue_size=1)
    representation_pub = rospy.Publisher('/deformation_representation', Image, queue_size=1)

    ref_ros = CvBridge().cv2_to_imgmsg(sensor.ref)
    ref_ros.header.stamp = rospy.Time.now()
    ref_ros.header.frame_id = 'ref'
    rectify_crop_ref_img_publisher.publish(ref_ros)

    while not rospy.is_shutdown():
        rectify_crop_image = sensor.get_rectify_crop_image()
        now_time = rospy.Time.now()
        cv2.imshow('rectify_crop_image', rectify_crop_image)
        rectify_crop_image_ros = CvBridge().cv2_to_imgmsg(rectify_crop_image)
        rectify_crop_image_ros.header.stamp = now_time
        rectify_crop_image_ros.header.frame_id = 'rectify_crop'
        rectify_crop_image_publisher.publish(rectify_crop_image_ros)

        representation, mixed_visualization = \
            sensor.raw_image_2_representation(cv2.cvtColor(rectify_crop_image, cv2.COLOR_BGR2GRAY))
        cv2.imshow('mixed_visualization', mixed_visualization)
        representation_ros = CvBridge().cv2_to_imgmsg(representation)
        representation_ros.header.stamp = now_time
        representation_ros.header.frame_id = 'representation'
        representation_pub.publish(representation_ros)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break












