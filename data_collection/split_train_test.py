import numpy as np
import glob
import random


root_path = '../Dataset'
test_number = 10000
raw_images = glob.glob(root_path + '/image/*/*.png')
print("Total raw images: ", len(raw_images))
random.shuffle(raw_images)
train_raw_images = np.array(raw_images[:-test_number])
test_raw_images = np.array(raw_images[-test_number::])
print("Num train raw images: ", len(train_raw_images))
print("Num test raw images: ", len(test_raw_images))
np.save(root_path + '/train_images.npy', train_raw_images)
np.save(root_path + '/test_images.npy', test_raw_images)

train_mixed_images = []
for image_path in train_raw_images:
    image_path_split = image_path.split('image')
    train_mixed_image = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    train_mixed_images.append(train_mixed_image)
print("Num train mixed images: ", len(train_mixed_images))
np.save(root_path + '/train_mixed_images.npy', np.array(train_mixed_images))

test_mixed_images = []
for image_path in test_raw_images:
    image_path_split = image_path.split('image')
    test_mixed_image = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    test_mixed_images.append(test_mixed_image)
print("Num test mixed images: ", len(test_mixed_images))
np.save(root_path + '/test_mixed_images.npy', np.array(test_mixed_images))

train_wrenches = []
for image_path in train_raw_images:
    image_path_split = image_path.split('image')
    train_wrench = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    train_wrenches.append(train_wrench)
print("Num train wrenches: ", len(train_wrenches))
np.save(root_path + '/train_wrench.npy', np.array(train_wrenches))

test_wrenches = []
for image_path in test_raw_images:
    image_path_split = image_path.split('image')
    test_wrench = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    test_wrenches.append(test_wrench)
print("Num test wrenches: ", len(test_wrenches))
np.save(root_path + '/test_wrench.npy', np.array(test_wrenches))


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
