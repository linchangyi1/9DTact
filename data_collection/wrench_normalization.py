import glob
import numpy as np
import os

root_path = '../Dataset'
wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.2, 0.2], [-0.2, 0.2], [-0.05, 0.05]]
min_wrench = np.array([.0, .0, .0, .0, .0, .0])
max_wrench = np.array([.0, .0, .0, .0, .0, .0])
for i in range(len(wrench_range)):
    min_wrench[i] = wrench_range[i][0]
    max_wrench[i] = wrench_range[i][1]
wrench_span = max_wrench - min_wrench

all_wrench = glob.glob(root_path + '/wrench/*/*.npy')
for wrench_path in all_wrench:
    if 'norm' in wrench_path:
        os.remove(wrench_path)

# judge whether there is any wrench beyond the range and remove the image and wrench
original_wrench = glob.glob(root_path + '/wrench/*/*.npy')
print('There are {} wrenches.'.format(len(original_wrench)))

original_images = glob.glob(root_path + '/image/*/*.png')
print('There are {} images.'.format(len(original_images)))

count = 0
for wrench_path in original_wrench:
    wrench = np.load(wrench_path)
    if sum((wrench - min_wrench) < 0) or sum((wrench - max_wrench) > 0):
        print(wrench_path)
        count += 1
        os.remove(wrench_path)
        split_path = wrench_path.split('wrench')
        image_postfix = split_path[-1].split('npy')
        image_path = split_path[0] + 'image' + image_postfix[0] + 'png'
        os.remove(image_path)
    # print(sum((wrench - min_wrench) < 0))
    # print(sum((wrench - max_wrench) > 0))
print('{} wrench beyond the range.'.format(count))


# find whether there is any image deleted
original_wrench = glob.glob(root_path + '/wrench/*/*.npy')
for wrench_path in original_wrench:
    split_path = wrench_path.split('wrench')
    image_postfix = split_path[-1].split('npy')
    image_path = split_path[0] + 'image' + image_postfix[0] + 'png'
    if not os.path.exists(image_path):
        os.remove(wrench_path)
        print('The image {} is removed, so the wrench {} is also being removed.'.format(image_path, wrench_path))
        # print(wrench_path)


# rename the images and wrenches to be correct order
for i in range(175):
    object_ID = i + 1
    images_path = sorted(glob.glob(root_path + '/image/' + str(object_ID) + '/*.png'), key=os.path.getmtime)
    # print(images_path)
    for j in range(len(images_path)):
        src_path = images_path[j]
        dst_path = root_path + '/image/' + str(object_ID) + '/' + str(j+1) + '.png'
        os.rename(src_path, dst_path)
        split_path = src_path.split('image')
        wrench_postfix = split_path[-1].split('png')
        src_path = split_path[0] + 'wrench' + wrench_postfix[0] + 'npy'
        dst_path = root_path + '/wrench/' + str(object_ID) + '/' + str(j + 1) + '.npy'
        os.rename(src_path, dst_path)

original_wrench = glob.glob(root_path + '/wrench/*/*.npy')
# save normalized wrench
for wrench_path in original_wrench:
    if 'norm' not in wrench_path:
        wrench = np.load(wrench_path)
        wrench_normalization = (wrench - min_wrench) / wrench_span
        new_path = wrench_path.split('.npy')[0] + '_norm.npy'
        np.save(new_path, wrench_normalization)


