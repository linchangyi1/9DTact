import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import cv2
import yaml
from shape_reconstruction import Sensor
from force_estimation import Estimator
from force_estimation import Visualizer

if __name__ == '__main__':
    shape_file = open("../shape_reconstruction/shape_config.yaml", 'r+', encoding='utf-8')
    shape_config = yaml.load(shape_file, Loader=yaml.FullLoader)
    sensor = Sensor(shape_config)

    force_file = open("force_config.yaml", 'r+', encoding='utf-8')
    force_config = yaml.load(force_file, Loader=yaml.FullLoader)
    estimator = Estimator(force_config)
    visualizer = Visualizer()

    while sensor.cap.isOpened():
        image = sensor.get_rectify_crop_image()
        cv2.imshow('image', image)
        representation, _ = sensor.raw_image_2_representation(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        cv2.imshow('representation', representation)
        force = estimator.predict_force(representation)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if not visualizer.vis.poll_events():
            break
        else:
            visualizer.update_force(force)



