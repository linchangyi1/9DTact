import cv2
import numpy as np
from torch.utils.data import Dataset


class DTactDataset(Dataset):
    def __init__(self, mode='train', root_path='../Dataset', image_type='RGB',
                 test_object=False, mixed_image=True) -> None:
        super().__init__()
        self.mode = mode
        self.image_type = image_type
        self.mixed_image = mixed_image
        if mode == 'train':
            if test_object:
                if mixed_image:
                    self.images = np.load(root_path + '/train_mixed_images_object.npy')
                else:
                    self.images = np.load(root_path + '/train_images_object.npy')
                self.wrenches = np.load(root_path + '/train_wrench_object.npy')
            else:
                if mixed_image:
                    self.images = np.load(root_path + '/train_mixed_images.npy')
                else:
                    self.images = np.load(root_path + '/train_images.npy')
                self.wrenches = np.load(root_path + '/train_wrench.npy')
        else:
            if test_object:
                if mixed_image:
                    self.images = np.load(root_path + '/test_mixed_images_object.npy')
                else:
                    self.images = np.load(root_path + '/test_images_object.npy')
                self.wrenches = np.load(root_path + '/test_wrench_object.npy')
            else:
                if mixed_image:
                    self.images = np.load(root_path + '/test_mixed_images.npy')
                else:
                    self.images = np.load(root_path + '/test_images.npy')
                self.wrenches = np.load(root_path + '/test_wrench.npy')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path).astype('float32')
        if self.image_type != 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.transpose([2, 0, 1])
        force_path = self.wrenches[index]
        force = np.load(force_path)
        if self.mode == 'train':
            return image, force
        else:
            return image, force, image_path


if __name__ == '__main__':
    dataset_path = '../Dataset'
    dataset = DTactDataset(mode='train', root_path=dataset_path)
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        print(dataset.__getitem__(i)[0])
