import numpy as np
import cv2
from shape_reconstruction import Camera
import yaml


class Sensor(Camera):
    def __init__(self, cfg, calibrated=True, ref=None):
        super().__init__(cfg)
        depth_calibration = cfg['depth_calibration']
        self.depth_calibration_dir = self.calibration_sensor_dir + \
            depth_calibration['depth_calibration_dir']
        self.Pixel_to_Depth_path = self.depth_calibration_dir + \
            depth_calibration['Pixel_to_Depth_path']

        if calibrated:
            self.Pixel_to_Depth = np.load(self.Pixel_to_Depth_path)
            self.max_index = len(self.Pixel_to_Depth) - 1

            # parameters for height_map
            if ref is None:
                self.ref = self.get_rectify_crop_avg_image()
            else:
                self.ref = ref
            self.ref_GRAY = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
            sensor_reconstruction = cfg['sensor_reconstruction']
            self.lighting_threshold = sensor_reconstruction['lighting_threshold']
            self.kernel_list = sensor_reconstruction['kernel_list']
            self.contact_gray_base = sensor_reconstruction['contact_gray_base']
            self.depth_k = sensor_reconstruction['depth_k']

            self.expand_x = int(28.0/self.pixel_per_mm) + 2
            self.expand_y = int(21.0/self.pixel_per_mm) + 2

            # parameters for point_cloud
            self.points = np.zeros([self.expand_x * self.expand_y, 3])
            self.X, self.Y = np.meshgrid(
                np.arange(self.expand_x), np.arange(self.expand_y))
            Z = np.zeros_like(self.X)
            self.points[:, 0] = np.ndarray.flatten(self.X) * self.pixel_per_mm
            self.points[:, 1] = -np.ndarray.flatten(self.Y) * self.pixel_per_mm
            self.points[:, 2] = np.ndarray.flatten(Z)

            self.mixed_image = np.zeros_like(self.ref)
            self.mixed_image[::, ::, 0] = self.ref_GRAY
            self.mixed_visualization = np.zeros_like(self.ref)
            self.mixed_visualization[::, ::, 0] = self.ref_GRAY

    def raw_image_2_height_map(self, img_GRAY):
        diff_raw = self.ref_GRAY - img_GRAY - self.lighting_threshold
        diff_mask = (diff_raw < 100).astype(np.uint8)
        diff = diff_raw * diff_mask + self.lighting_threshold
        diff[diff > self.max_index] = self.max_index
        diff = cv2.GaussianBlur(diff.astype(np.float32), (7, 7), 0).astype(int)
        height_map = self.Pixel_to_Depth[diff] - \
            self.Pixel_to_Depth[self.lighting_threshold]
        for kernel in self.kernel_list:
            height_map = cv2.GaussianBlur(
                height_map.astype(np.float32), (kernel, kernel), 0)
        return height_map

    def raw_image_2_representation(self, img_GRAY):
        diff_darker = self.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
        diff_darker[diff_darker < 0] = 0
        diff_brighter = img_GRAY.astype(np.float32) - self.ref_GRAY.astype(np.float32)
        diff_brighter[diff_brighter < 0] = 0
        scale = 3
        self.mixed_image[::, ::, 1] = diff_brighter * scale
        self.mixed_image[::, ::, 2] = diff_darker * scale

        # mixed_visualization is for better showing the gel flow with less image noise
        diff_darker[diff_darker < 3] = 0
        diff_brighter[diff_brighter < 3] = 0
        self.mixed_visualization[::, ::, 1] = diff_brighter * 7
        self.mixed_visualization[::, ::, 2] = diff_darker * 5
        return self.mixed_image, self.mixed_visualization

    def visualize_gel_deformation(self, img):
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff_dark = self.ref_GRAY.astype(
            np.float32) - img_GRAY.astype(np.float32)
        vis_img = np.ones(
            (img_GRAY.shape[0], img_GRAY.shape[1], 3), dtype=np.uint8) - 1
        threshold = 4
        showing_scale = 255 // max(diff_dark.max(), -diff_dark.min())
        vis_img[diff_dark > threshold,
                2] = diff_dark[diff_dark > threshold] * showing_scale
        vis_img[diff_dark < -threshold, 1] = - \
            diff_dark[diff_dark < -threshold] * showing_scale
        return vis_img

    def get_height_map(self):
        img = self.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.flip(self.get_rectify_crop_image(), 1)
        height_map = self.raw_image_2_height_map(img_GRAY)
        return height_map

    def height_map_2_depth_map(self, height_map):
        contact_show = np.zeros_like(height_map)
        contact_show[height_map > 0] = self.contact_gray_base
        depth_map = height_map * self.depth_k + contact_show
        depth_map = depth_map.astype(np.uint8)
        return depth_map

    def height_2_depth(self, height_map):
        depth_map = height_map * (self.depth_k+10)
        depth_map = depth_map.astype(np.uint8)
        return depth_map

    def get_depth_map(self):
        height_map = self.get_height_map()
        depth_map = self.height_map_2_depth_map(height_map)
        return depth_map

    def height_map_2_point_cloud(self, height_map):
        self.points[:, 2] = np.ndarray.flatten(height_map)
        # self.points[:, 2] = - np.ndarray.flatten(height_map)
        return self.points

    def height_map_2_point_cloud_gradients(self, height_map):
        height_gradients = np.gradient(height_map)
        points = self.height_map_2_point_cloud(height_map)
        return points, height_gradients

    def get_point_cloud(self):
        height_map = self.get_height_map()
        points = self.height_map_2_point_cloud(height_map)
        return points

    def get_point_cloud_gradients(self):
        height_map = self.get_height_map()
        points, height_gradients = self.height_map_2_point_cloud_gradients(
            height_map)
        return points, height_gradients

    def expand_image(self, img):
        img_expand = np.zeros([self.expand_y, self.expand_x])
        img_expand[int((self.expand_y - img.shape[0]) / 2):int((self.expand_y + img.shape[0]) / 2),
                   int((self.expand_x - img.shape[1]) / 2):int((self.expand_x + img.shape[1]) / 2)] = img

        return img_expand


if __name__ == '__main__':
    f = open("shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor = Sensor(cfg)
    while sensor.cap.isOpened():
        img = sensor.get_rectify_crop_image()
        cv2.imshow('RawImage', img)
        vis_img = sensor.visualize_gel_deformation(img)
        cv2.imshow('vis_img', vis_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
