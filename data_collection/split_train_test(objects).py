import numpy as np
import glob
import random

object_number = 175
select_number = 2
select_object = random.sample(range(1, object_number + 1), select_number)
print(select_object)

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
print(len(train_save))
print(len(test_save))
print(test_list)
np.save(root_path + '/train_images_object.npy', train_save)
np.save(root_path + '/test_images_object.npy', test_save)

mixed_image_paths = []
for image_path in train_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    mixed_image_paths.append(mixed_image_path)
mixed_image_paths = np.array(mixed_image_paths)
np.save(root_path + '/train_mixed_images_object.npy', mixed_image_paths)

mixed_image_paths = []
for image_path in test_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    mixed_image_path = image_path_split[0] + 'mixed_image' + image_path_split[-1]
    mixed_image_paths.append(mixed_image_path)
mixed_image_paths = np.array(mixed_image_paths)
np.save(root_path + '/test_mixed_images_object.npy', mixed_image_paths)

wrench_paths = []
for image_path in train_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    force_path = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    wrench_paths.append(force_path)
wrench_paths = np.array(wrench_paths)
np.save(root_path + '/train_wrench_object.npy', wrench_paths)

wrench_paths = []
for image_path in test_save:
    # print(image_path)
    image_path_split = image_path.split('image')
    force_path = str(image_path_split[0] + 'wrench' + image_path_split[1].split('.')[0] + '_norm.npy')
    wrench_paths.append(force_path)
wrench_paths = np.array(wrench_paths)
np.save(root_path + '/test_wrench_object.npy', wrench_paths)






