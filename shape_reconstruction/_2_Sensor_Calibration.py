# collect images and calibrate the sensor
import numpy as np
import cv2
import yaml
import os
from shape_reconstruction import Sensor


class SensorCalibration:
    def __init__(self, cfg_path):
        f = open(cfg_path, 'r+', encoding='utf-8')
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.sensor = Sensor(cfg, calibrated=False)
        if not os.path.exists(self.sensor.depth_calibration_dir):
            os.makedirs(self.sensor.depth_calibration_dir)
        depth_calibration = cfg['depth_calibration']
        self.BallRad = depth_calibration['BallRad']
        self.circle_detection_gray = depth_calibration['circle_detect_gray']
        self.show_circle_detection = depth_calibration['show_circle_detection']

    def run(self):
        print("DON'T touch the sensor surface!!!!!")
        print('Please press "y" to save the reference image!')
        ref = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.sensor.depth_calibration_dir + '/ref.png', ref)
        rc_ref = self.sensor.rectify_crop_image(ref)
        cv2.imwrite(self.sensor.depth_calibration_dir + '/rectify_crop_ref.png', rc_ref)
        print('Reference image saved!')
        print('Please press the ball on the sensor and press "y" to save the sample image!')
        sample = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.sensor.depth_calibration_dir + '/sample.png', sample)
        rc_sample = self.sensor.rectify_crop_image(sample)
        cv2.imwrite(self.sensor.depth_calibration_dir + '/rectify_crop_sample.png', rc_sample)
        print('Sample image saved!')

        gray_list, depth_list = self.mapping_data_collection(rc_sample, rc_ref)
        gray_list = np.array(gray_list)
        depth_list = np.array(depth_list)
        Pixel_to_Depth = self.get_list(gray_list, depth_list)
        np.save(self.sensor.Pixel_to_Depth_path, Pixel_to_Depth)

    def circle_detection(self, diff):
        diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
        contact_mask = (diff_gray > self.circle_detection_gray).astype(np.uint8)
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        if len(sorted_areas):
            cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if self.show_circle_detection:
                key = -1
                print('If the detected circle is suitable, press the key "q" to continue!')
                while key != ord('q'):
                    center = (int(x), int(y))
                    radius = int(radius)
                    circle_show = cv2.circle(np.array(diff), center, radius, (0, 255, 0), 1)
                    circle_show[int(y), int(x)] = [255, 255, 255]
                    cv2.imshow('contact', circle_show.astype(np.uint8))
                    key = cv2.waitKey(0)
                    if key == ord('w'):
                        y -= 1
                    elif key == ord('s'):
                        y += 1
                    elif key == ord('a'):
                        x -= 1
                    elif key == ord('d'):
                        x += 1
                    elif key == ord('m'):
                        radius += 1
                    elif key == ord('n'):
                        radius -= 1
                cv2.destroyWindow('contact')
            return center, radius
        else:
            return (0, 0), 0

    def mapping_data_collection(self, img, ref):
        gray_list = []
        depth_list = []
        diff_raw = ref - img
        diff_mask = (diff_raw < 150).astype(np.uint8)
        diff = diff_raw * diff_mask
        cv2.imshow('ref', ref)
        cv2.imshow('img', img)
        cv2.imshow('diff', diff)
        center, detect_radius_p = self.circle_detection(diff)
        if detect_radius_p:
            x = np.linspace(0, diff.shape[0] - 1, diff.shape[0])  # [0, 479]
            y = np.linspace(0, diff.shape[1] - 1, diff.shape[1])  # [0, 639]
            xv, yv = np.meshgrid(y, x)
            xv = xv - center[0]
            yv = yv - center[1]
            rv = np.sqrt(xv ** 2 + yv ** 2)
            mask = (rv < detect_radius_p)
            temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.sensor.pixel_per_mm ** 2
            height_map = (np.sqrt(self.BallRad ** 2 - temp) * mask - np.sqrt(
                self.BallRad ** 2 - (detect_radius_p * self.sensor.pixel_per_mm) ** 2)) * mask
            height_map[np.isnan(height_map)] = 0
            diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
            # diff_gray = self.sensor.crop_image(diff_gray)
            # height_map = self.sensor.crop_image(height_map)
            count = 0
            for i in range(height_map.shape[0]):
                for j in range(height_map.shape[1]):
                    if height_map[i, j] > 0:
                        gray_list.append(diff_gray[i, j])
                        depth_list.append(height_map[i, j])
                        count += 1
            print('Sample points number: {}'.format(count))
            return gray_list, depth_list

    def get_list(self, gray_list, depth_list):
        GRAY_scope = int(gray_list.max())
        GRAY_Height_list = np.zeros(GRAY_scope + 1)
        for gray_number in range(GRAY_scope + 1):
            gray_height_sum = depth_list[gray_list == gray_number].sum()
            gray_height_num = (gray_list == gray_number).sum()
            if gray_height_num:
                GRAY_Height_list[gray_number] = gray_height_sum / gray_height_num
        for gray_number in range(GRAY_scope + 1):
            if GRAY_Height_list[gray_number] == 0:
                if not gray_number:
                    min_index = gray_number - 1
                    max_index = gray_number + 1
                    for i in range(GRAY_scope - gray_number):
                        if GRAY_Height_list[gray_number + 1 + i] != 0:
                            max_index = gray_number + 1 + i
                            break
                    GRAY_Height_list[gray_number] = (GRAY_Height_list[max_index] - GRAY_Height_list[min_index]) / (
                            max_index - min_index)
        return GRAY_Height_list


if __name__ == '__main__':
    config_path = 'shape_config.yaml'
    depth_calibration = SensorCalibration(config_path)
    depth_calibration.run()
