# -*- coding: utf-8 -*-
# Sun Xiaofei
import argparse
import os
import sys
import tensorflow as tf
sys.path.extend([os.path.abspath("."), os.path.abspath("./../..")])
from ccyclegan_model import ccyclegan
import inout_util_mat as ut
os.chdir(os.getcwd() + '/..')
print('pwd : {}'.format(os.getcwd()))
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='')
# -------------------------------------
#set load directory
parser.add_argument('--data_path', dest='mat_path', default= 'data', help='file directory')
parser.add_argument('--LRUS_path', dest='LRUS_path', default= 'LR', help='LRUS image folder name')
parser.add_argument('--HRUS_path', dest='HRUS_path', default= 'HR', help='HRUS image folder name')
parser.add_argument('--LRUS_val_path', dest='LRUS_val_path', default= 'LR_val', help='LRUS val image folder name')
parser.add_argument('--HRUS_val_path', dest='HRUS_val_path', default= 'HR_val', help='HRUS val image folder name')
parser.add_argument('--LRUS_test_path', dest='LRUS_test_path', default= 'LR_test', help='LRUS test image folder name')
parser.add_argument('--HRUS_test_path', dest='HRUS_test_path', default= 'HR_test', help='HRUS test image folder name')
parser.add_argument('--data_info', dest='data_info', type=ut.ParseList, default='US')

#set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',  default='checkpoint', help='check point dir')
parser.add_argument('--test_npy_save_dir', dest='test_npy_save_dir',  default='test', help='test numpy file save dir')

#image info
parser.add_argument('--patch_size', dest='patch_size', type=int,  default=56, help='image patch size, h=w')
parser.add_argument('--whole_size', dest='whole_size', type=int,  default=256, help='image whole size, h=w')
parser.add_argument('--img_channel', dest='img_channel', type=int,  default=1, help='image channel, 1')
parser.add_argument('--img_vmax', dest='img_vmax', type=int, default=0, help='max value in image')
parser.add_argument('--img_vmin', dest='img_vmin', type=int, default=-60,  help='max value in image')

#train, test
parser.add_argument('--model', dest='model', default='ccyclegan', help='cyclegan, ccyclegan')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

#train detail
parser.add_argument('--end_epoch', dest='end_epoch', type=int, default=200, help='end epoch')
parser.add_argument('--decay_epoch', dest='decay_epoch', type=int, default=100, help='epoch to decay lr')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int,  default=2, help='batch size')
parser.add_argument('--L1_lambda1', dest='L1_lambda1', type=float, default=10.0, help='weight of cyclic loss')
parser.add_argument('--L1_lambda2', dest='L1_lambda2', type=float, default=5.0, help='weight of identical loss')
parser.add_argument('--L1_lambda3', dest='L1_lambda3', type=float, default=5.0, help='weight of cro-coe loss')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='The exponential decay rate for the 1st moment estimates.')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='The exponential decay rate for the 2nd moment estimates.')
parser.add_argument('--ngf', dest='ngf', type=int, default=256, help='# of gen filters in first conv layer')
parser.add_argument('--nglf', dest='nglf', type=int, default=15, help='# of gen filters in last conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')

#others
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10, help='save a model every save_freq (iteration)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=2, help='print_freq (iterations)') #default
parser.add_argument('--continue_train', dest='continue_train', type=ut.ParseBoolean, default=True, help='load the latest model: true, false')
parser.add_argument('--gpu_no', dest='gpu_no', type=int,  default=0, help='gpu no')
parser.add_argument('--unpair', dest='unpair', type=ut.ParseBoolean, default=True, help='unpaired image(cycle loss) : True|False')

# -------------------------------------
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
tf.compat.v1.disable_eager_execution()
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tfconfig)
model = ccyclegan(sess, args)
model.train(args) if args.phase == 'train' else model.test(args)