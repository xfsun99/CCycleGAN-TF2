# -*- coding: utf-8 -*-
# Sun Xiaofei
from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import ccyclegan_module as md
import inout_util_mat as ut
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class ccyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess

        ####modality folder name
        self.data_info = args.data_info

        # save directory
        self.p_info = '_'.join(self.data_info)
        self.checkpoint_dir = os.path.join('.', args.checkpoint_dir, self.p_info)
        self.log_dir = os.path.join('.', 'logs', self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(self.checkpoint_dir, self.log_dir))

        # module
        self.discriminator = md.discriminator
        self.generator = md.generator

        # network options
        OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
        self.options = OPTIONS._make((args.ngf, args.nglf, args.ndf,
                                      args.img_channel, args.phase == 'train'))

        """
        load images
        """
        print('data load... ultrasound -> numpy')
        self.image_loader = ut.USDataLoader(args.mat_path, args.LRUS_path, args.HRUS_path, \
                                             image_size=args.whole_size, patch_size=args.patch_size,
                                             depth=args.img_channel,
                                             image_max=args.img_vmax, image_min=args.img_vmin,
                                             batch_size=args.batch_size, \
                                             is_unpair=args.unpair, model=args.model)

        self.val_image_loader = ut.USDataLoader(args.mat_path, args.LRUS_val_path, args.HRUS_val_path, \
                                                 image_size=args.whole_size, patch_size=args.patch_size,
                                                 depth=args.img_channel,
                                                 image_max=args.img_vmax, image_min=args.img_vmin,
                                                 batch_size=args.batch_size, \
                                                 is_unpair=args.unpair, model=args.model)

        self.test_image_loader = ut.USDataLoader(args.mat_path, args.LRUS_test_path, args.HRUS_test_path, \
                                                  image_size=args.whole_size, patch_size=args.patch_size,
                                                  depth=args.img_channel,
                                                  image_max=args.img_vmax, image_min=args.img_vmin,
                                                  batch_size=args.batch_size, \
                                                  is_unpair=args.unpair, model=args.model)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader()
            self.val_image_loader()
            print('Data load complete !!!, Elapsed time: {}\nN_train : {}, N_val: {}'.format(time.time() - t1,
                                                                                 len(self.image_loader.LRUS_image_name),
                                                                                 len(self.val_image_loader.LRUS_image_name)))
            [self.patch_A, self.patch_B] = self.image_loader.input_pipeline(self.sess, args.patch_size, args.end_epoch)
        else:
            self.test_image_loader()
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1,
                                                                   len(self.test_image_loader.LRUS_image_name)))
            self.patch_A = tf.compat.v1.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel],
                                          name='LRUS')
            self.patch_B = tf.compat.v1.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel],
                                          name='HRUS')

        """
        build model
        """
        #### image placehold(for validation)
        self.val_A = tf.compat.v1.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='val_A')
        self.val_B = tf.compat.v1.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='val_B')
        #### image placehold(for test)
        self.test_A = tf.compat.v1.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='test_A')
        self.test_B = tf.compat.v1.placeholder(tf.float32, [None, args.whole_size, args.whole_size, args.img_channel], name='test__B')

        #### Generator & Discriminator
        # Generator
        self.G_A = self.generator(self.patch_A, self.options, False, name="generatorA2B")
        self.F_GA = self.generator(self.G_A, self.options, False, name="generatorB2A")
        self.F_B = self.generator(self.patch_B, self.options, True, name="generatorB2A")
        self.G_FB = self.generator(self.F_B, self.options, True, name="generatorA2B")

        self.G_B = self.generator(self.patch_B, self.options, True, name="generatorA2B")  # G : x->y
        self.F_A = self.generator(self.patch_A, self.options, True, name="generatorB2A")  # F : y->X

        # Discriminator
        self.D_GA = self.discriminator(self.G_A, self.options, reuse=False, name="discriminatorY")
        self.D_FB = self.discriminator(self.F_B, self.options, reuse=False, name="discriminatorX")
        self.D_B = self.discriminator(self.patch_B, self.options, reuse=True, name="discriminatorY")
        self.D_A = self.discriminator(self.patch_A, self.options, reuse=True, name="discriminatorX")

        ####Reserved for test
        self.test_G_A = self.generator(self.test_A, self.options, True, name="generatorA2B")

        #### Loss
        # generator loss
        self.cycle_loss = md.cycle_loss(self.patch_A, self.F_GA, self.patch_B, self.G_FB, args.L1_lambda1)
        self.identical_loss = md.identical_loss(self.patch_A, self.G_B, self.patch_B, self.F_A, args.L1_lambda2)
        self.G_loss_A2B = md.least_square(self.D_GA, tf.ones_like(self.D_GA))
        self.G_loss_B2A = md.least_square(self.D_FB, tf.ones_like(self.D_FB))
        self.cor_coe_loss_A = md.cor_coe_loss(self.F_GA, self.patch_A, args.L1_lambda3)
        self.cor_coe_loss_B = md.cor_coe_loss(self.G_FB, self.patch_B, args.L1_lambda3)
        self.G_loss = self.G_loss_A2B + self.G_loss_B2A + self.cycle_loss + self.identical_loss + self.cor_coe_loss_A + self.cor_coe_loss_B

        # dicriminator loss
        self.D_loss_patch_B = md.least_square(self.D_B, tf.ones_like(self.D_B))
        self.D_loss_patch_GA = md.least_square(self.D_GA, tf.zeros_like(self.D_GA))
        self.D_loss_patch_A = md.least_square(self.D_A, tf.ones_like(self.D_A))
        self.D_loss_patch_FB = md.least_square(self.D_FB, tf.zeros_like(self.D_FB))

        self.D_loss_B = (self.D_loss_patch_B + self.D_loss_patch_GA)
        self.D_loss_A = (self.D_loss_patch_A + self.D_loss_patch_FB)
        self.D_loss = (self.D_loss_B + self.D_loss_A) / 2

        #### variable list
        t_vars = tf.compat.v1.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        #### optimizer
        self.lr = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.g_optim = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)

        """
        Summary
        """
        #### loss summary
        # generator
        self.G_loss_sum = tf.compat.v1.summary.scalar("1_G_loss", self.G_loss, family='Generator_loss')
        self.cycle_loss_sum = tf.compat.v1.summary.scalar("2_cycle_loss", self.cycle_loss, family='Generator_loss')
        self.identical_loss_sum = tf.compat.v1.summary.scalar("3_identical_loss", self.identical_loss, family='Generator_loss')
        self.G_loss_A2B_sum = tf.compat.v1.summary.scalar("4_G_loss_A2B", self.G_loss_A2B, family='Generator_loss')
        self.G_loss_B2A_sum = tf.compat.v1.summary.scalar("5_G_loss_B2A", self.G_loss_B2A, family='Generator_loss')
        self.g_sum = tf.compat.v1.summary.merge([self.G_loss_sum, self.cycle_loss_sum, self.identical_loss_sum, self.G_loss_A2B_sum, self.G_loss_B2A_sum])

        # discriminator
        self.D_loss_sum = tf.compat.v1.summary.scalar("1_D_loss", self.D_loss, family='Discriminator_loss')
        self.D_loss_B_sum = tf.compat.v1.summary.scalar("2_D_loss_B", self.D_loss_patch_B, family='Discriminator_loss')
        self.D_loss_GA_sum = tf.compat.v1.summary.scalar("3_D_loss_GA", self.D_loss_patch_GA, family='Discriminator_loss')
        self.d_sum = tf.compat.v1.summary.merge([self.D_loss_sum, self.D_loss_B_sum, self.D_loss_GA_sum])

        #### image summary
        self.val_G_A = self.generator(self.val_A, self.options, True, name="generatorA2B")
        self.val_G_B = self.generator(self.val_B, self.options, True, name="generatorB2A")
        self.train_img_summary = tf.concat([self.patch_A, self.patch_B, self.G_A], axis=2)
        self.summary_image_1 = tf.compat.v1.summary.image('1_train_patch_image', self.train_img_summary)
        self.val_img_summary = tf.concat([self.val_A, self.val_B, self.val_G_A], axis=2)
        self.summary_image_2 = tf.compat.v1.summary.image('2_val_whole_image', self.val_img_summary)


        #### psnr summary
        self.summary_psnr_LRUS = tf.compat.v1.summary.scalar("1_psnr_LRUS", ut.tf_psnr(self.val_A, self.val_B, 2),
                                                   family='PSNR')  # -1 ~ 1
        self.summary_psnr_result = tf.compat.v1.summary.scalar("2_psnr_output", ut.tf_psnr(self.val_B, self.val_G_A, 2),
                                                     family='PSNR')  # -1 ~ 1
        self.summary_psnr = tf.compat.v1.summary.merge([self.summary_psnr_LRUS, self.summary_psnr_result])

        # model saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        print('--------------------------------------------\n# of parameters : {} '. \
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

    def train(self, args):
        init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        self.sess.run(init_op)
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir, self.sess.graph)

        # pretrained model load
        self.start_step = 0  # load SUCCESS -> self.start_step // failed -> 0
        if args.continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # iteration -> epoch
        self.start_epoch = int((self.start_step + 1) / len(self.image_loader.LRUS_image_name))

        print('Start point : iter : {}, epoch : {}'.format(self.start_step, self.start_epoch))

        start_time = time.time()
        lr = args.lr
        for epoch in range(self.start_epoch, args.end_epoch):

            batch_idxs = len(self.image_loader.LRUS_image_name)

            # decay learning rate
            if epoch > args.decay_epoch:
                lr = args.lr - (epoch - (args.decay_epoch)) * ((args.lr / (args.end_epoch - args.decay_epoch)))

            for _ in range(0, batch_idxs):
                # Update G network
                A, B, F_B, G_A, _, G_loss, summary_str = self.sess.run(
                    [self.patch_A, self.patch_B, self.F_B, self.G_A, self.g_optim, self.G_loss, self.g_sum], feed_dict={self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)

                # Update D network
                _, D_loss, summary_str = self.sess.run(
                    [self.d_optim, self.D_loss, self.d_sum], feed_dict={self.patch_A: A, self.patch_B: B, self.lr: lr})

                self.writer.add_summary(summary_str, self.start_step)

                if (self.start_step + 1) % args.print_freq == 0:
                    currt_step = self.start_step % len(
                        self.image_loader.LRUS_image_name) if epoch != 0 else self.start_step
                    print(("Epoch: {} {}/{} time: {} lr:{}, Gen_loss: {}, Dis_loss: {}".format(epoch, currt_step, batch_idxs,
                                                                     time.time() - start_time, lr, G_loss, D_loss)))
                    # summary training sample image
                    summary_str1 = self.sess.run(self.summary_image_1)
                    self.writer.add_summary(summary_str1, self.start_step)

                    # check validation sample image
                    self.check_validation(args, self.start_step, direction='A2B')
                    self.check_validation(args, self.start_step, direction='B2A')

                if (self.start_step + 1) % args.save_freq == 0:
                    self.save(args, self.start_step)

                self.start_step += 1

        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(self.image_loader.enqueue_threads)

    # summary validation sample image during training
    def check_validation(self, args, idx, direction):
        sltd_idx = np.random.choice(range(len(self.val_image_loader.LRUS_image_name)))

        sample_A_image, sample_B_image = self.val_image_loader.LRUS_images[sltd_idx], \
                                         self.val_image_loader.HRUS_images[sltd_idx]

        if direction == 'A2B':
            G_A = self.sess.run(
                self.val_G_A,
                feed_dict={self.val_B: sample_B_image.reshape([1] + self.val_B.get_shape().as_list()[1:]),
                           self.val_A: sample_A_image.reshape([1] + self.val_A.get_shape().as_list()[1:])})
            G_A = np.array(G_A).astype(np.float32)

            summary_str1, summary_str2 = self.sess.run(
                [self.summary_image_2, self.summary_psnr],
                feed_dict={self.val_A: sample_A_image.reshape([1] + self.val_A.get_shape().as_list()[1:]),
                           self.val_B: sample_B_image.reshape([1] + self.val_B.get_shape().as_list()[1:]),
                           self.val_G_A: G_A.reshape([1] + self.val_G_A.get_shape().as_list()[1:])})

            self.writer.add_summary(summary_str1, idx)
            self.writer.add_summary(summary_str2, idx)

        elif direction == 'B2A':
            G_B = self.sess.run(
                self.val_G_B,
                feed_dict={self.val_A: sample_A_image.reshape([1] + self.val_A.get_shape().as_list()[1:]),
                           self.val_B: sample_B_image.reshape([1] + self.val_B.get_shape().as_list()[1:])})
            G_B = np.array(G_B).astype(np.float32)

            summary_str1, summary_str2 = self.sess.run(
                [self.summary_image_2, self.summary_psnr],
                feed_dict={self.val_B: sample_B_image.reshape([1] + self.val_B.get_shape().as_list()[1:]),
                           self.val_A: sample_A_image.reshape([1] + self.val_A.get_shape().as_list()[1:]),
                           self.val_G_B: G_B.reshape([1] + self.val_G_B.get_shape().as_list()[1:])})

            self.writer.add_summary(summary_str1, idx)
            self.writer.add_summary(summary_str2, idx)

    # save model
    def save(self, args, step):
        model_name = args.model + ".model"
        self.checkpoint_dir = os.path.join('.', self.checkpoint_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    # load model
    def load(self):
        print(" [*] Reading checkpoint...")
        self.checkpoint_dir = os.path.join('.', self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS for Test Data")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)
        npy_save_dir = os.path.join('.', args.test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)

        ## test
        for idx in range(len(self.test_image_loader.LRUS_images)):

            test_A = self.test_image_loader.LRUS_images[idx]

            mk_G_A = self.sess.run(self.test_G_A,
                                   feed_dict={self.test_A: test_A.reshape([1] + self.test_A.get_shape().as_list()[1:])})

            save_file_nm_g = 'Gen_from_' + self.test_image_loader.LRUS_image_name[idx]

            np.save(os.path.join(npy_save_dir, save_file_nm_g), mk_G_A)
