import random
import sys
import tty
import os
import cv2
import ros_numpy
import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from rokubimini_msgs.srv import ResetWrench, ResetWrenchRequest
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image

IMAGE_DIR = "../Dataset/image/"
MIXED_IMAGE_DIR = "../Dataset/mixed_image/"
WRENCH_DIR = "../Dataset/wrench/"
PRESS_THRESHOLD = -0.3
RANDOM_RANGE = -3.0
STATE_INFO_INTERVAL = rospy.Duration(1)

wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.3, 0.3], [-0.3, 0.3], [-0.05, 0.05]]

min_wrench = np.array([.0, .0, .0, .0, .0, .0])
max_wrench = np.array([.0, .0, .0, .0, .0, .0])
for i in range(len(wrench_range)):
    min_wrench[i] = wrench_range[i][0]
    max_wrench[i] = wrench_range[i][1]

judge_threshold = 12
desired_number = 4

judge_scale = []
for i in range(len(wrench_range)):
    judge_scale.append(judge_threshold / ((wrench_range[i][1] - wrench_range[i][0]) / desired_number))
judge_scale[3] = 0
judge_scale[4] = 0
judge_scale = np.array(judge_scale)


class DataNode(object):
    def __init__(self):
        rospy.init_node("recoder")

        # Recording information
        self.object_ID = 1
        self.count = 0
        self.count_last = 0
        self.pressing = False
        self.random_threshold = 0
        self.lastinfo_time = rospy.Time.now() - STATE_INFO_INTERVAL

        # Reset wrench
        reset_service = '/bus0/ft_sensor0/reset_wrench'
        rospy.wait_for_service(reset_service)
        self.resetting = False
        self.reset_wrench_srv = rospy.ServiceProxy(reset_service, ResetWrench)
        self.reset_request = ResetWrenchRequest()
        self.reset_request.desired_wrench.force.x = 0
        self.reset_request.desired_wrench.force.y = 0
        self.reset_request.desired_wrench.force.z = 0
        self.reset_request.desired_wrench.torque.x = 0
        self.reset_request.desired_wrench.torque.y = 0
        self.reset_request.desired_wrench.torque.z = 0
        self.reset_sensor()

        # Topics to subscribe
        ref_image_topic = "/rectify_crop_ref_image"
        ref = ros_numpy.numpify(rospy.wait_for_message(ref_image_topic, Image))
        self.ref_GRAY = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self.mixed_image = np.zeros_like(ref)
        self.mixed_image[::, ::, 0] = self.ref_GRAY
        image_topic = "/rectify_crop_image"
        wrench_topic = "/bus0/ft_sensor0/ft_sensor_readings/wrench"
        rospy.wait_for_message(wrench_topic, WrenchStamped)
        image_sub = Subscriber(image_topic, Image)
        wrench_sub = Subscriber(wrench_topic, WrenchStamped)
        sync = ApproximateTimeSynchronizer([image_sub, wrench_sub], 10, 0.01)
        sync.registerCallback(self.data_callback)

        # Save data
        self.saved_wrench = []
        self.image_sub_dir = IMAGE_DIR + str(self.object_ID)
        self.mixed_image_sub_dir = MIXED_IMAGE_DIR + str(self.object_ID)
        self.wrench_sub_dir = WRENCH_DIR + str(self.object_ID)
        if not os.path.exists(self.image_sub_dir):
            os.makedirs(self.image_sub_dir)
        if not os.path.exists(self.mixed_image_sub_dir):
            os.makedirs(self.mixed_image_sub_dir)
        if not os.path.exists(self.wrench_sub_dir):
            os.makedirs(self.wrench_sub_dir)

    def range_judge(self, wrench):
        wrench = np.array(wrench)
        beyond = sum((wrench - min_wrench) < 0) or sum((wrench - max_wrench) > 0)
        return not beyond

    # Judge whether to save the data or not
    def judge(self, wrench):
        wrench_list = np.array(self.saved_wrench)
        wrench = np.array(wrench)
        wrench_diff = wrench_list - wrench
        wrench_diff_scaled = np.abs(wrench_diff) * judge_scale
        diff = np.sum(wrench_diff_scaled, axis=1)
        min_diff = np.min(diff)

        return min_diff > judge_threshold

    def data_callback(self, image_msg, wrench_msg):
        # rospy.loginfo('receive data')
        if not self.resetting:
            wrench = [wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z,
                      wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z]
            # under pressing
            if wrench[2] < PRESS_THRESHOLD:
                # a new geometry contacts with sensor
                if not self.pressing:
                    self.random_threshold = random.uniform(RANDOM_RANGE, 0)
                    # save the first data if f_z is smaller a random threshold
                    if wrench[2] < PRESS_THRESHOLD + self.random_threshold and self.range_judge(wrench):
                        self.saved_wrench.append(wrench)
                        image = ros_numpy.numpify(image_msg)
                        self.save_data(image, wrench)
                        self.count_last = self.count
                        self.pressing = True
                # still in the same pressing process
                elif self.judge(wrench) and self.range_judge(wrench):
                    self.saved_wrench.append(wrench)
                    image = ros_numpy.numpify(image_msg)
                    self.save_data(image, wrench)
            else:
                self.pressing = False
                self.saved_wrench = []

    def reset_sensor(self):
        rospy.logwarn("\r\nSensor will be reset!\r\n")
        self.resetting = True
        reset_feedback = self.reset_wrench_srv.call(self.reset_request)
        self.resetting = False
        if reset_feedback:
            rospy.loginfo(self.reset_request)
            rospy.loginfo("Reset successfully!")
        else:
            rospy.loginfo("Fail to reset!")

    def run(self):
        tty.setraw(sys.stdin.fileno())
        rospy.loginfo("Start to collect the %dth object.\r\n" % self.object_ID)
        while not rospy.is_shutdown():
            key = sys.stdin.read(1)
            if key == "n":  # next object
                self.object_ID = self.object_ID + 1
                self.image_sub_dir = IMAGE_DIR + str(self.object_ID)
                self.mixed_image_sub_dir = MIXED_IMAGE_DIR + str(self.object_ID)
                self.wrench_sub_dir = WRENCH_DIR + str(self.object_ID)
                if not os.path.exists(self.image_sub_dir):
                    os.makedirs(self.image_sub_dir)
                if not os.path.exists(self.mixed_image_sub_dir):
                    os.makedirs(self.mixed_image_sub_dir)
                if not os.path.exists(self.wrench_sub_dir):
                    os.makedirs(self.wrench_sub_dir)
                self.count = 0
                rospy.loginfo("Start to collect the %dth object.\r\n" % self.object_ID)
                if os.path.exists(self.image_sub_dir + "/1.png"):
                    rospy.logwarn("Please check whether you need to change the object ID!!!")
            elif key == "z":
                self.count = self.count_last
                rospy.logwarn("Clear last pressing!\r\n")
            elif key == "s":
                self.count = 0
                rospy.logwarn("Clear all pressings before!\r\n")
            elif key == "r":
                self.reset_sensor()
            elif key == "q":
                quit()

    def save_data(self, img, wrench):
        self.count = self.count + 1
        prefix = str(self.count)
        sample_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff_darker = self.ref_GRAY.astype(np.float32) - sample_GRAY.astype(np.float32)
        diff_darker[diff_darker < 0] = 0
        diff_brighter = sample_GRAY.astype(np.float32) - self.ref_GRAY.astype(np.float32)
        diff_brighter[diff_brighter < 0] = 0
        scale = 3
        self.mixed_image[::, ::, 1] = diff_brighter * scale
        self.mixed_image[::, ::, 2] = diff_darker * scale
        image_name = self.image_sub_dir + '/' + prefix + ".png"
        mixed_image_name = self.mixed_image_sub_dir + '/' + prefix + ".png"
        wrench_name = self.wrench_sub_dir + '/' + prefix + ".npy"
        cv2.imwrite(image_name, img)
        cv2.imwrite(mixed_image_name, self.mixed_image)
        np.save(wrench_name, wrench)
        rospy.loginfo("Saved the latest data as %s and %s.\r\n" % (image_name, wrench_name))


if __name__ == '__main__':
    node = DataNode()
    node.run()
