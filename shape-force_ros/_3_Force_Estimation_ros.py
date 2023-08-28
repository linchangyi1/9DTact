import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import rospy
import yaml
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped
import ros_numpy
from force_estimation import Estimator


class ForceROS:
    def __init__(self, cfg):
        rospy.init_node("force_ros")
        self.force = np.array([.5, .5, 1.0, .5, .5, .5])
        self.estimator = Estimator(cfg)
        self.predicted_wrench_pub = rospy.Publisher('/predicted_wrench', WrenchStamped, queue_size=1)
        self.force_msg = WrenchStamped()
        self.rate = rospy.Rate(90)
        self.representation_sub = rospy.Subscriber('/deformation_representation', Image, self.predict_callback)

    def predict_callback(self, image_msg):
        representation = ros_numpy.numpify(image_msg)
        self.force = self.estimator.predict_force(representation)
        self.force_msg.wrench.force.x = self.force[0]
        self.force_msg.wrench.force.y = self.force[1]
        self.force_msg.wrench.force.z = self.force[2]
        self.force_msg.wrench.torque.x = self.force[3]
        self.force_msg.wrench.torque.y = self.force[4]
        self.force_msg.wrench.torque.z = self.force[5]
        self.predicted_wrench_pub.publish(self.force_msg)

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == '__main__':
    f = open("../force_estimation/force_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    force_ros = ForceROS(cfg)
    force_ros.run()
