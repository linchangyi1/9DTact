import numpy as np
import glob
import random

object_number = 175
select_number = 18
select_object = random.sample(range(1, object_number + 1), select_number)
print('Selected objects: ', select_object)

train_list = []
test_list = []

root_path = '../Dataset'
for object_id in range(object_number):
    object_id += 1
    images = glob.glob(root_path + '/image/' + str(object_id) + '/*.png')
    if object_id in select_object:
        test_list += images
    else:
        train_list += images

random.shuffle(train_list)
train_save = np.array(train_list)
test_save = np.array(test_list)
print("Num train images: ", len(train_save))
print("Num test images: ", len(test_save))
np.save(root_path + '/train_images_object.npy', train_save)
np.save(root_path + '/test_images_object.npy', test_save)

train_mixed_images = []
for image_path in train_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    train_mixed_image = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    train_mixed_images.append(train_mixed_image)
np.save(root_path + '/train_mixed_images_object.npy', np.array(train_mixed_images))

test_mixed_images = []
for image_path in test_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    test_mixed_image = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    test_mixed_images.append(test_mixed_image)
np.save(root_path + '/test_mixed_images_object.npy', np.array(test_mixed_images))

train_wrenches = []
for image_path in train_save:
    image_path_split = image_path.split('image')
    train_wrench = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    train_wrenches.append(train_wrench)
np.save(root_path + '/train_wrench_object.npy', np.array(train_wrenches))

test_wrenches = []
for image_path in test_save:
    image_path_split = image_path.split('image')
    test_wrench = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    test_wrenches.append(test_wrench)
np.save(root_path + '/test_wrench_object.npy', np.array(test_wrenches))






