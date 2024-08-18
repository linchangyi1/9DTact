import numpy as np
import cv2
import yaml


class Camera:
    def __init__(self, cfg, calibrated=True):
        sensor_id = cfg['sensor_id']
        camera_setting = cfg['camera_setting']
        camera_channel = camera_setting['camera_channel']
        raw_img_width = camera_setting['resolution'][0]
        raw_img_height = camera_setting['resolution'][1]
        fps = camera_setting['fps']
        self.cap = cv2.VideoCapture(camera_channel)
        if self.cap.isOpened():
            print('------Camera is open--------')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, raw_img_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, raw_img_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        calibration_root_dir = cfg['calibration_root_dir']
        self.calibration_sensor_dir = calibration_root_dir + '/sensor_' + str(sensor_id)
        camera_calibration = cfg['camera_calibration']
        self.camera_calibration_dir = self.calibration_sensor_dir + camera_calibration['camera_calibration_dir']
        self.row_index_path = self.camera_calibration_dir + camera_calibration['row_index_path']
        self.col_index_path = self.camera_calibration_dir + camera_calibration['col_index_path']
        self.position_scale_path = self.camera_calibration_dir + camera_calibration['position_scale_path']

        self.crop_img_height = camera_calibration['crop_size'][0]
        self.crop_img_width = camera_calibration['crop_size'][1]

        if calibrated:
            self.row_index = np.load(self.row_index_path)
            self.col_index = np.load(self.col_index_path)
            position_scale = np.load(self.position_scale_path)
            center_position = position_scale[0:2]
            self.pixel_per_mm = position_scale[2]
            self.height_begin = int(center_position[0] - self.crop_img_height / 2)
            self.height_end = int(center_position[0] + self.crop_img_height / 2)
            self.width_begin = int(center_position[1] - self.crop_img_width / 2)
            self.width_end = int(center_position[1] + self.crop_img_width / 2)

    def get_raw_image(self):
        return self.cap.read()[1]

    def rectify_image(self, img):
        img_rectify = img[self.row_index, self.col_index]
        return img_rectify

    def crop_image(self, img):
        return img[self.height_begin:self.height_end, self.width_begin:self.width_end]

    def rectify_crop_image(self, img):
        img = self.crop_image(self.rectify_image(img))
        return img

    def get_rectify_image(self):
        img = self.rectify_image(self.get_raw_image())
        return img

    def get_rectify_crop_image(self):
        img = self.crop_image(self.get_rectify_image())
        return img

    def get_raw_avg_image(self):
        global img
        while True:
            img = self.cap.read()[1]
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.cap.read()[1]
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_rectify_avg_image(self):
        global img
        while True:
            img = self.get_rectify_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_image()
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_rectify_crop_avg_image(self):
        global img
        while True:
            img = self.get_rectify_crop_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_crop_image()
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def img_list_avg_rectify(self, img_list):
        img_1 = cv2.imread(img_list[0])
        img_add = np.zeros_like(img_1, float)
        for img_path in img_list:
            img = cv2.imread(img_path)
            img_add += img
        img_avg = img_add / len(img_list)
        img_avg = img_avg.astype(np.uint8)
        ref_img_avg = self.rectify_image(img_avg)
        return ref_img_avg


if __name__ == '__main__':
    f = open("shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    camera = Camera(cfg)
    while True:
        raw_img = camera.get_raw_image()
        cv2.imshow('raw_img', raw_img)
        raw_img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('raw_img_GRAY', raw_img_GRAY)
        rectify_img = camera.get_rectify_image()
        cv2.imshow('rectify_img', rectify_img)
        rectify_crop_img = camera.get_rectify_crop_image()
        cv2.imshow('rectify_crop_img', rectify_crop_img)
        rectify_crop_GRAY = cv2.cvtColor(rectify_crop_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('rectify_crop_GRAY', rectify_crop_GRAY)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
