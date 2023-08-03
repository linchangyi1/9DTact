import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import cv2
import yaml
import rospy
from sensor_msgs.msg import Image
import ros_numpy
from shape_reconstruction import Sensor
from shape_visualizer import Visualizer


class ShapeROS:
    def __init__(self, cfg):
        rospy.init_node("shape_ros")

        # Topics to subscribe
        ref_image_topic = "/rectify_crop_ref_image"
        ref = ros_numpy.numpify(rospy.wait_for_message(ref_image_topic, Image))
        print("Receive the reference image.")
        self.sensor = Sensor(cfg, ref=ref)
        self.visualizer = Visualizer(self.sensor.points)
        self.height_map = self.map(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))

        image_topic = "/rectify_crop_image"
        self.img_sub = rospy.Subscriber(image_topic, Image, self.reconstruction)

    def map(self, img_GRAY):
        height_map = self.sensor.raw_image_2_height_map(img_GRAY)
        height_map = self.sensor.expand_image(height_map)
        return height_map

    def reconstruction(self, image_msg):
        img = ros_numpy.numpify(image_msg)
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.height_map = self.map(img_GRAY)

    def run(self):
        while not rospy.is_shutdown():
            if self.visualizer.vis.poll_events():
                points, gradients = self.sensor.height_map_2_point_cloud_gradients(
                    self.height_map)
                self.visualizer.update(points, gradients)
            else:
                break


if __name__ == '__main__':
    f = open("ros_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    shape_ros = ShapeROS(cfg)
    shape_ros.run()

