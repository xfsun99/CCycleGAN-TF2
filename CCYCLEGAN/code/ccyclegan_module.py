# -*- coding: utf-8 -*-
# Sun Xiaofei
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def discriminator(image, options, reuse=False, name='discriminator'):
    def first_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
        with tf.compat.v1.variable_scope(name):
            return leakyrelu(conv2d(input_, out_channels, ks=ks, s=s))

    def conv_layer(input_, out_channels, ks=3, s=1, name='disc_conv_layer'):
        with tf.compat.v1.variable_scope(name):
            return leakyrelu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))

    def last_layer(input_, out_channels, ks=4, s=1, name='disc_conv_layer'):
        with tf.compat.v1.variable_scope(name):
            return tf.keras.layers.Dense( out_channels)(conv2d(input_, out_channels, ks=ks, s=s))

    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        l1 = first_layer(image, options.df_dim, ks=4, s=2, name='disc_layer1')
        l2 = conv_layer(l1, options.df_dim * 2, ks=4, s=2, name='disc_layer2')
        l3 = conv_layer(l2, options.df_dim * 4, ks=4, s=2, name='disc_layer3')
        l4 = conv_layer(l3, options.df_dim * 8, ks=4, s=1, name='disc_layer4')
        l5 = last_layer(l4, options.img_channel, ks=4, s=1, name='disc_layer5')
        return l5


def generator(image, options, reuse=False, name="generator"):
    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False

        def conv_layer(input_, out_channels, ks=3, s=1, name='conv_layer'):
            with tf.compat.v1.variable_scope(name):
                return tf.nn.relu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))

        def gen_module(input_, out_channels, ks=3, s=1, name='gen_module'):
            with tf.compat.v1.variable_scope(name):
                ml1 = conv_layer(input_, out_channels, ks, s, name=name + '_l1')
                ml2 = conv_layer(ml1, out_channels, ks, s, name=name + '_l2')
                ml3 = conv_layer(ml2, out_channels, ks, s, name=name + '_l3')
                concat_l = input_ + ml3
                m_out = tf.nn.relu(concat_l)
                return m_out

        l1 = conv_layer(image, options.gf_dim, name='convlayer1')
        module1 = gen_module(l1, options.gf_dim, name='gen_module1')
        module2 = gen_module(module1, options.gf_dim, name='gen_module2')
        module3 = gen_module(module2, options.gf_dim, name='gen_module3')
        module4 = gen_module(module3, options.gf_dim, name='gen_module4')
        module5 = gen_module(module4, options.gf_dim, name='gen_module5')
        concate_layer = tf.concat([l1, module1, \
                                   module2, module3, module4, module5], axis=3, name='concat_layer')
        concat_conv_l1 = conv_layer(concate_layer, options.gf_dim, ks=3, s=1, name='concat_convlayer1')
        last_conv_layer = conv_layer(concat_conv_l1, options.glf_dim, ks=3, s=1, name='last_conv_layer')
        output = tf.add(conv2d(last_conv_layer, options.img_channel, ks=3, s=1), image, name='output')
        return output

    # network components

def leakyrelu(x, leak=0.2, name="leakyrelu"):
    with tf.compat.v1.variable_scope(name):
        return tf.maximum(x, leak * x)


def batchnorm(input_, name="batch_norm"):
    with tf.compat.v1.variable_scope(name):
        return tf.compat.v1.layers.batch_normalization(input_, axis=3, epsilon=1e-5, \
                                             momentum=0.1, training=True, \
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def conv2d(batch_input, out_channels, ks=4, s=2, name="cov2d"):
    with tf.compat.v1.variable_scope(name):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.compat.v1.layers.conv2d(padded_input, out_channels, kernel_size=ks, \
                                strides=s, padding="valid", \
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))


#### loss
def least_square(A, B):
    return tf.reduce_mean((A - B) ** 2)


def cycle_loss(A, F_GA, B, G_FB, lambda_):
    return lambda_ * (tf.reduce_mean(tf.abs(A - F_GA)) + tf.reduce_mean(tf.abs(B - G_FB)))


def identical_loss(A, G_B, B, F_A, gamma):
    return gamma * (tf.reduce_mean(tf.abs(G_B - B)) + tf.reduce_mean(tf.abs(F_A - A)))

def cor_coe_loss(A,B,delta):
    A_var=A-tf.reduce_mean(A)
    B_var=B-tf.reduce_mean(B)
    r_num=tf.reduce_sum(A_var*B_var)
    r_den=tf.sqrt(tf.reduce_sum(A_var**2))*tf.sqrt(tf.reduce_sum(B_var**2))
    r=r_num/r_den
    return delta * (1-r**2)