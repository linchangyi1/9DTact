# collect images and calibrate the camera
import numpy as np
import cv2
import yaml
import os
from shape_reconstruction import Camera
from scipy.interpolate import Rbf


class CameraCalibration:
    def __init__(self, cfg_path):
        f = open(cfg_path, 'r+', encoding='utf-8')
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.camera = Camera(cfg, calibrated=False)

        camera_calibration = cfg['camera_calibration']
        self.row_points = camera_calibration['row_points']
        self.col_points = camera_calibration['col_points']
        self.grid_distance = camera_calibration['grid_distance']
        if not os.path.exists(self.camera.camera_calibration_dir):
            os.makedirs(self.camera.camera_calibration_dir)
        self.image_format = camera_calibration['image_format']

    def run(self):
        # record reference and sample images
        print("DON'T touch the sensor surface!!!!!")
        print('Please press "y" to save the reference image!')
        ref = self.camera.get_raw_avg_image()
        cv2.imwrite(self.camera.camera_calibration_dir + '/ref.' + self.image_format, ref)
        print('Reference image saved!')
        print('Please press the calibration board on the sensor and press "y" to save the sample image!')
        while True:
            sample = self.camera.get_raw_image()
            cv2.imshow('sample', sample)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.imwrite(self.camera.camera_calibration_dir + '/sample.' + self.image_format, sample)
                cv2.destroyWindow('sample')
                break
            if key == ord('q'):
                quit()
        self.calibrate_image(ref, sample)

    def calibrate_image(self, ref, sample):
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        sample_contours = sample.copy()
        diff = ref - sample
        diff_mask = (diff < 100).astype(np.uint8)
        diff = diff * diff_mask
        diff[diff < 5] = 0  # change this threshold for your sensor
        cv2.imshow('diff', diff)
        binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
        cv2.imshow('binary', binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('morph', morph)
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sample_contours, contours, -1, (0, 255, 0), 3)
        cv2.imshow('contours', sample_contours)

        sample_drawing = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        command_color = (0, 0, 0)
        special_point_color = (255, 0, 0)
        special_lines_color = (0, 0, 255)
        point_size = 6
        line_size = 2
        all_point = []
        j = 0
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            # if the area is too big or too small, ignore it
            if cv2.contourArea(contours[i]) < 200 or cv2.contourArea(contours[i]) > 2000:
                continue
            j += 1
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            all_point.append([cy, cx])

        all_point = np.array(all_point)
        # sort according to the row value
        all_point = all_point[np.lexsort(all_point[:, ::-1].T)]
        # sort according to the column value for each row
        for i in range(self.row_points):
            temp = all_point[i * self.col_points:(i + 1) * self.col_points]
            all_point[i * self.col_points:(i + 1) * self.col_points] = temp[np.lexsort(temp.T)]

        # draw lines as a mesh grid
        for i in range(self.row_points):
            for j in range(self.col_points-1):
                cv2.line(sample_drawing, (all_point[i*self.col_points+j][1], all_point[i*self.col_points+j][0]),
                         (all_point[i*self.col_points+j + 1][1], all_point[i*self.col_points+j + 1][0]), command_color,
                         line_size)
        for i in range(self.col_points):
            for j in range(self.row_points-1):
                cv2.line(sample_drawing, (all_point[j*self.col_points+i][1], all_point[j*self.col_points+i][0]),
                         (all_point[(j+1)*self.col_points+i][1], all_point[(j+1)*self.col_points+i][0]), command_color,
                         line_size)

        # calculate the average pixel distance (sample the four edges near the central point)
        center_index = (self.row_points * self.col_points) // 2
        dis_sum = 0
        for around_index in (-self.col_points, -1, 1, self.col_points):
            dis_sum += np.linalg.norm(all_point[center_index] - all_point[center_index + around_index], ord=2)
            cv2.line(sample_drawing, (all_point[center_index][1], all_point[center_index][0]),
                     (all_point[center_index + around_index][1], all_point[center_index + around_index][0]),
                     special_lines_color, line_size+2)
        dis_avg = dis_sum / 4

        # save the position of the center point and the pixel to mm ratio
        position_scale = all_point[center_index].tolist()  # center_position
        pixel_per_mm = float(self.grid_distance / dis_avg)  # pixel_per_mm
        position_scale.append(pixel_per_mm)
        print(position_scale)
        np.save(self.camera.position_scale_path, position_scale)

        # draw the points
        for point in all_point:
            cv2.circle(sample_drawing, (point[1], point[0]), point_size, command_color, -1)
        # draw the central point with special color
        for around_index in (-self.col_points, -1, 0, 1, self.col_points):
            cv2.circle(sample_drawing, (all_point[center_index + around_index][1],
                                        all_point[center_index + around_index][0]), point_size + 4,
                       special_point_color, -1)
        cv2.imshow('sample_drawing', sample_drawing)

        # calculate the real position for the sampling points
        init_position = all_point.copy()
        real_position = np.zeros_like(init_position)
        for i in range(self.row_points):
            for j in range(self.col_points):
                real_position[i*self.col_points+j] = init_position[center_index] +\
                            dis_avg * np.array([i - self.row_points // 2, j - self.col_points // 2])

        # get the row and column indexes for correcting distortion
        itp_row = Rbf(real_position[::, 0], real_position[::, 1], init_position[::, 0], function='cubic')
        itp_col = Rbf(real_position[::, 0], real_position[::, 1], init_position[::, 1], function='cubic')
        col_mesh, row_mesh = np.meshgrid(range(ref.shape[1]), range(ref.shape[0]))
        row_index = itp_row(row_mesh, col_mesh).astype(np.int32)
        col_index = itp_col(row_mesh, col_mesh).astype(np.int32)

        # make sure no outside index
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if row_index[i, j] < 0 or row_index[i, j] > 479 or col_index[i, j] < 0 or col_index[i, j] > 639:
                    row_index[i, j] = 0
                    col_index[i, j] = 0

        # save the index
        np.save(self.camera.row_index_path, row_index)
        np.save(self.camera.col_index_path, col_index)
        cv2.waitKey()

        # correct the sample image and show it
        sample_new = sample[row_index, col_index]
        cv2.imshow('sample_new', sample_new)
        height_begin = int(all_point[center_index][0] - self.camera.crop_img_height / 2)
        height_end = int(all_point[center_index][0] + self.camera.crop_img_height / 2)
        width_begin = int(all_point[center_index][1] - self.camera.crop_img_width / 2)
        width_end = int(all_point[center_index][1] + self.camera.crop_img_width / 2)
        sample_new_crop = sample_new[height_begin:height_end, width_begin:width_end]
        cv2.imshow('sample_new_crop', sample_new_crop)
        cv2.imwrite(self.camera.camera_calibration_dir + '/ref_GRAY.' + self.image_format, ref)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_GRAY.' + self.image_format, sample)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_contours.' + self.image_format, sample_contours)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_drawing.' + self.image_format, sample_drawing)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_new.' + self.image_format, sample_new)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_new_crop.' + self.image_format, sample_new_crop)
        sample_drawing_new = sample_drawing[row_index, col_index]
        sample_drawing_new_crop = sample_drawing_new[height_begin:height_end, width_begin:width_end]
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_drawing_new.' + self.image_format, sample_drawing_new)
        cv2.imwrite(self.camera.camera_calibration_dir + '/sample_drawing_new_crop.' + self.image_format,
                    sample_drawing_new_crop)
        cv2.waitKey()


if __name__ == '__main__':
    config_path = 'shape_config.yaml'
    cc = CameraCalibration(config_path)
    cc.run()
