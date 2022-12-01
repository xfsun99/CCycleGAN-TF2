# -*- coding: utf-8 -*-
# Sun Xiaofei
import os
from glob import glob
import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from random import shuffle
import threading
import scipy.io as scio
import numpy as np
import cv2

class USDataLoader(object):
    def __init__(self, mat_path, LRUS_image_path, HRUS_image_path, \
                 image_size=256, patch_size=56, depth=1, \
                 image_max=0, image_min=-60, batch_size=1, \
                 is_unpair=False, model='', num_threads=1, extension='mat'):

        # file dir
        self.extension = extension
        self.mat_path = mat_path
        self.LRUS_image_path = LRUS_image_path
        self.HRUS_image_path = HRUS_image_path

        # image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth

        self.image_max = image_max
        self.image_min = image_min

        # training params
        self.batch_size = batch_size
        self.is_unpair = is_unpair
        self.model = model

        # slice name
        self.LRUS_image_name, self.HRUS_image_name = [], []

        # batch generator parameters
        self.num_threads = num_threads
        self.capacity = 20 * self.num_threads * self.batch_size
        self.min_queue = 10 * self.num_threads * self.batch_size

    # mat file -> numpy array
    def __call__(self):
        p_LRUS = []
        p_HRUS = []
        resize_img = 256
        #print(os.path.join(self.mat_path, self.LRUS_image_path, '*.' + self.extension))
        p_LRUS_path, p_HRUS_path = \
            glob(os.path.join(self.mat_path, self.LRUS_image_path, '*.' + self.extension)), \
            glob(os.path.join(self.mat_path, self.HRUS_image_path, '*.' + self.extension))
        org_LRUS_images, LRUS_slice_no = self.load_scan(p_LRUS_path, resize_img)
        org_HRUS_images, HRUS_slice_no = self.load_scan(p_HRUS_path, resize_img)
        # print(LRUS_slice_nm)
        # print(len(org_LRUS_images))\

        self.LRUS_image_name.extend(LRUS_slice_no)
        self.HRUS_image_name.extend(HRUS_slice_no)
        org_LRUS_images=np.array(org_LRUS_images)
        org_HRUS_images=np.array(org_HRUS_images)
        # normalization
        p_LRUS.append(self.normalize(org_LRUS_images, self.image_max, self.image_min))
        p_HRUS.append(self.normalize(org_HRUS_images, self.image_max, self.image_min))

        self.LRUS_images = np.concatenate(tuple(p_LRUS), axis=0)
        self.HRUS_images = np.concatenate(tuple(p_HRUS), axis=0)

        # image index
        self.LRUS_index, self.HRUS_index = list(range(len(self.LRUS_images))), list(range(len(self.HRUS_images)))

    def load_scan(self, path, resize_img):
        # print(path)
        org_images = []
        image_no = []
        num = 0
        for s in path:
            data = scio.loadmat(s)
            imgs = np.array(data['norm_binary'])
            imgs = imgs.transpose([2, 1, 0])
            for i in range(imgs.shape[0]):
                img = imgs[i].transpose([1, 0])
                img = cv2.resize(img, (resize_img, resize_img))
                org_images.append(img)
                num += 1
                image_no.append(str(num))

        return org_images, image_no

    def normalize(self, img, max_=0, min_=-60):
        img = img.astype(np.float32)
        if 'cycle' in self.model:  # 0 ~ 1
            img = (img - min_) / (max_ - min_)
            return img
        else:  # -1 ~ 1
            img = 2 * ((img - min_) / (max_ - min_)) - 1
            return img

    # CCycleGAN
    def augumentation(self, LRUS, HRUS):
        """
        sltd_random_indx[0] :
            1: rotation
            2. flipping
            3. scaling
            4. pass
        sltd_random_indx[1] :
            select params
        """
        sltd_random_indx = [np.random.choice(range(4)), np.random.choice(range(2))]
        if sltd_random_indx[0] == 0:
            return rotate(LRUS, 45, reshape=False), rotate(HRUS, 45, reshape=False)
        elif sltd_random_indx[0] == 1:
            param = [True, False][sltd_random_indx[1]]
            if param:
                return LRUS[:, ::-1], HRUS[:, ::-1]  # horizontal
            return LRUS[::-1, :], HRUS[::-1, :]  # vertical
        elif sltd_random_indx[0] == 2:
            param = [0.5, 2][sltd_random_indx[1]]
            return LRUS * param, HRUS * param
        elif sltd_random_indx[0] == 3:
            return LRUS, HRUS

    def get_random_patches(self, LRUS_slice, HRUS_slice, patch_size, whole_size = 256):
        whole_h = whole_w = whole_size
        h = w = patch_size

        # patch image range
        hd, hu = h // 2, int(whole_h - np.round(h / 2))
        wd, wu = w // 2, int(whole_w - np.round(w / 2))

        # patch image center(coordinate on whole image)
        h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))
        LRUS_patch = LRUS_slice[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
        HRUS_patch = HRUS_slice[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]

        if self.model.lower() == 'ccyclegan':
            return self.augumentation(LRUS_patch, HRUS_patch)
        return LRUS_patch, HRUS_patch

    def input_pipeline(self, sess, image_size, end_point, depth=1):

        queue_input = tf.compat.v1.placeholder(tf.float32)
        queue_output = tf.compat.v1.placeholder(tf.float32)
        queue = tf.compat.v1.FIFOQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32], \
                             shapes=[(image_size, image_size, depth), (image_size, image_size, depth)])
        enqueue_op = queue.enqueue_many([queue_input, queue_output])
        close_op = queue.close()
        dequeue_op = queue.dequeue_many(self.batch_size)

        def enqueue(coord):
            enqueue_size = max(200, self.batch_size)
            self.step = 0
            while not coord.should_stop():
                LRUS_imgs, HRUS_imgs = [], []
                for i in range(enqueue_size):
                    if self.is_unpair:
                        L_sltd_idx = np.random.choice(self.LRUS_index)
                        H_sltd_idx = np.random.choice(self.HRUS_index)
                    else:
                        L_sltd_idx = H_sltd_idx = np.random.choice(self.LRUS_index)

                    pat_LRUS, pat_HRUS = \
                        self.get_random_patches(self.LRUS_images[L_sltd_idx],
                                                self.HRUS_images[H_sltd_idx], image_size)
                    LRUS_imgs.append(np.expand_dims(pat_LRUS, axis=-1))
                    HRUS_imgs.append(np.expand_dims(pat_HRUS, axis=-1))

                sess.run(enqueue_op, feed_dict={queue_input: np.array(LRUS_imgs, dtype=object),
                                                queue_output: np.array(HRUS_imgs, dtype=object)})
                self.step += 1
            if self.step > end_point:
                coord.request_stop()
            sess.run(close_op)

        self.coord = tf.train.Coordinator()
        self.enqueue_threads = [threading.Thread(target=enqueue, args=(self.coord,)) for i in range(self.num_threads)]
        for t in self.enqueue_threads: t.start()

        return dequeue_op

# ROI crop
def ROI_img(whole_image, row=[100, 250], col=[75, 225]):
    patch_ = whole_image[row[0]:row[1], col[0]: col[1]]
    return np.array(patch_)

# psnr
def log10(x):
    numerator = tf.compat.v1.log(x)
    denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX=1.0):
    mse = tf.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))


def psnr(img1, img2, PIXEL_MAX=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# save mk img
def save_image(LRUS, HRUS, output_, save_dir='.', max_=1, min_=0):
    f, axes = plt.subplots(2, 3, figsize=(30, 20))

    axes[0, 0].imshow(LRUS, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 1].imshow(HRUS, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 2].imshow(output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[1, 0].imshow(HRUS.astype(np.float32) - LRUS.astype(np.float32), cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 1].imshow(HRUS - output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 2].imshow(output_ - LRUS, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0, 0].title.set_text('LRUS image')
    axes[0, 1].title.set_text('HRUS image')
    axes[0, 2].title.set_text('output image')

    axes[1, 0].title.set_text('HRUS - LRUS  image')
    axes[1, 1].title.set_text('HRUS - outupt image')
    axes[1, 2].title.set_text('output - LRUS  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()

# argparser string -> boolean type
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


# argparser string -> boolean type
def ParseList(l):
    return l.split(',')
