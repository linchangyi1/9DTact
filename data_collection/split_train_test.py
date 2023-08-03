import numpy as np
import glob
import random


# used for RGB and NOT MIXED
root_path = '../Dataset'
test_number = 1000
images = glob.glob(root_path + '/image/*/*.png')
print(len(images))
random.shuffle(images)
train_save = np.array(images[:-test_number])
test_save = np.array(images[-test_number::])
print(len(train_save))
print(len(test_save))
np.save(root_path + '/train_images.npy', train_save)
np.save(root_path + '/test_images.npy', test_save)

mixed_image_paths = []
for image_path in train_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    mixed_image_paths.append(mixed_image_path)
mixed_image_paths = np.array(mixed_image_paths)
np.save(root_path + '/train_mixed_images.npy', mixed_image_paths)

mixed_image_paths = []
for image_path in test_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    mixed_image_paths.append(mixed_image_path)
mixed_image_paths = np.array(mixed_image_paths)
np.save(root_path + '/test_mixed_images.npy', mixed_image_paths)

wrench_paths = []
for image_path in train_save:
    image_path_split = image_path.split('image')
    force_path = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    wrench_paths.append(force_path)
wrench_paths = np.array(wrench_paths)
np.save(root_path + '/train_wrench.npy', wrench_paths)


wrench_paths = []
for image_path in test_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    force_path = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    wrench_paths.append(force_path)
wrench_paths = np.array(wrench_paths)
np.save(root_path + '/test_wrench.npy', wrench_paths)




def copy_mixed_path():
    image_paths = np.load('../Dataset/train_images.npy')
    mixed_image_paths = []
    for image_path in image_paths:
        # print(image_path)
        image_path_split = image_path.split('image')
        mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
        mixed_image_paths.append(mixed_image_path)
    mixed_image_paths = np.array(mixed_image_paths)
    np.save('../Dataset/train_mixed_images.npy', mixed_image_paths)
    print(mixed_image_paths.shape)

    image_paths = np.load('../Dataset/test_images.npy')
    mixed_image_paths = []
    for image_path in image_paths:
        # print(image_path)
        image_path_split = image_path.split('image')
        mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
        mixed_image_paths.append(mixed_image_path)
    mixed_image_paths = np.array(mixed_image_paths)
    np.save('../Dataset/test_mixed_images.npy', mixed_image_paths)
    print(mixed_image_paths.shape)

    image_paths = np.load('../Dataset/train_images_object.npy')
    mixed_image_paths = []
    for image_path in image_paths:
        # print(image_path)
        image_path_split = image_path.split('image')
        mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
        mixed_image_paths.append(mixed_image_path)
    mixed_image_paths = np.array(mixed_image_paths)
    np.save('../Dataset/train_mixed_images_object.npy', mixed_image_paths)
    print(mixed_image_paths.shape)

    image_paths = np.load('../Dataset/test_images_object.npy')
    mixed_image_paths = []
    for image_path in image_paths:
        # print(image_path)
        image_path_split = image_path.split('image')
        mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
        mixed_image_paths.append(mixed_image_path)
    mixed_image_paths = np.array(mixed_image_paths)
    np.save('../Dataset/test_mixed_images_object.npy', mixed_image_paths)
    print(mixed_image_paths.shape)