#!/bin/bash/env 

from __future__ import print_function
from datetime import datetime
import os
import time
import cv2
import sys
import argparse
import glob
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from utils.utils import *
from LIP_model import *


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

parser = argparse.ArgumentParser(description = "classify clothes")
parser.add_argument('--input_dir' , default='./' , type=str, help="dir of original images")
parser.add_argument('--output_dir' , default='./output',type=str,help="output dir for results")
parser.add_argument('--buffer_size', default=500)
parser.add_argument('--interval_size', default=20)
parser.add_argument('--label_file' , default=1)
parser.add_argument('--pattern', '-p',action='store_true',
                    help='create output for pattern detection script to read from')
args = parser.parse_args()

N_CLASSES = 20
# was 384
INPUT_SIZE = (384, 384)

#number of IMAGES in dir
DATA_LIST_PATH_ROOT = 'LIP_JPPNet/datasets/examples/list/'
RESTORE_FROM = 'LIP_JPPNet/checkpoint'
LABELS = ['top', 'bottom', 'full']

def main():
    """Create the model and start the evaluation process."""
    DATA_LIST_PATH=os.path.join(DATA_LIST_PATH_ROOT, 'val'+str(args.label_file)+'.txt')
    
    pattern_output_dir=''
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for label in LABELS:
        label_output = os.path.join(args.output_dir, label)
        if not os.path.exists(label_output):
            os.mkdir(label_output)
    if args.pattern:
        pattern_output_dir = args.output_dir + '_pattern'
        if not os.path.exists(pattern_output_dir):
            os.makedirs(pattern_output_dir)
        for label in LABELS:
            label_output = os.path.join(pattern_output_dir, label)
            if not os.path.exists(label_output):
                os.mkdir(label_output)
                
    # if you want just the filenames sorted, simply remove the dir from each
    interval_size = int(args.interval_size)
    buffer_end = int(args.buffer_size)
    buffer_start = buffer_end-interval_size

    # sort the directory contents by why they were created in the dir
    # image paths are relatove paths !!
    DATA_DIRECTORY = args.input_dir
    image_list = os.listdir(DATA_DIRECTORY)
    full_list = [os.path.join(DATA_DIRECTORY, i) for i in image_list]
    # sort files by time created
    time_sorted_list = sorted(full_list, key=os.path.getctime)
    # relative sorted path
    sorted_filename_list = [ os.path.basename(i) for i in time_sorted_list]
    # only get images up to the buffer size
    DATA_LIST = sorted_filename_list[buffer_start :buffer_end]
    NUM_STEPS = 0
    
    # create label file for data loader and get number of IMAGES
    # image is concatenated with DATA_DIRECTORY in ImageReader init function!
    with open(DATA_LIST_PATH, 'w+') as f:
        for image in DATA_LIST:
            if image.split('.')[-1] in ['jpg','png','jpeg']:
                f.write('/'+image+'\n')
                NUM_STEPS += 1
                
    print("YOU HAVE THIS MANY ACTUAL IMAGES ",NUM_STEPS)
    if NUM_STEPS==0:
        print("Exiting Since no Images Found")
        return -1
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False,
                             coord, buffer_end, int(args.interval_size))
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

    
    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    #print(image_list , "     ", len(image_list))
    print("Prediciting Now")
    # Iterate over training steps.
    for step in range(NUM_STEPS):
        parsing_ = sess.run(pred_all)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print(image_list[step])
        img_split = image_list[step].split('/')
        #TODO save this 
        img_id , img_ext = img_split[-1].split('.')
        img_path = os.path.join(DATA_DIRECTORY, img_id+".{}".format(img_ext))
        
        msk = crop_images(img_path, parsing_)
        if msk.size != 0:
            for cropped_img, class_idx in msk:
                label = LABELS[class_idx]
                parsing_im = Image.fromarray(cropped_img)
                color_output = os.path.join(args.output_dir,label,img_id)
                parsing_im.save('{}_vis.{}'.format(color_output,img_ext))
                if args.pattern:
                    pattern_output = os.path.join(pattern_output_dir,label,img_id)
                    parsing_im.save('{}_vis.{}'.format(pattern_output,img_ext))
                #cv2.imwrite('{}/{}/{}.png'.format(args.output_dir, label, img_id), cropped_img)
        # for producing segmentation image
        #msk = decode_labels(parsing_, num_classes=N_CLASSES)
        #parsing_im = Image.fromarray(msk[0])
        #parsing_im.save('{}/{}_vis.png'.format(OUTPUT_DIR, img_id))
        #cv2.imwrite('{}/{}.png'.format(OUTPUT_DIR, img_id), parsing_[0,:,:,0])

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()


##############################################################333
