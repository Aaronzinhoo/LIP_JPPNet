from __future__ import print_function
import os
import sys
import argparse

from pathlib import Path
import tensorflow as tf

from utils.utils import *
from LIP_model import *
from utils.image_reader import ImageReader
from utils.model import JPPNetModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

parser = argparse.ArgumentParser(description="classify clothes as top, bottom, or full")
parser.add_argument('input_dir', default='./', type=str, help="dir of original images")
parser.add_argument('output_dir', default='./output',type=str,help="output dir for results")
parser.add_argument('--buffer_size', default=500,help='current size of dir')
parser.add_argument('--interval_size', default=20,help='max size that jpp should process')
parser.add_argument('--label_file', default=1)
parser.add_argument('--classes', default="fashion",choices=['fashion','lip'],help="Classes decide which labeling to use for the predictions and directory heirarchy")
parser.add_argument('--pattern', '-p',action='store_true',
                    help='create output for pattern detection script to read from')
parser.add_argument('--save_images', '-s', action='store_false',
                    help='store original images in dir')
parser.add_argument('--style_preprocess', action='store_true',
                    help='crop bbox of objects instead of segmenting for style classification')
args = parser.parse_args()

N_CLASSES = 20
INPUT_SIZE = (384, 384)

#number of IMAGES in dir
BASE =  Path(__file__).resolve().parent
DATA_LIST_PATH= str(BASE / 'datasets' / 'list' / 'val{}.txt'.format(args.label_file))
RESTORE_FROM = str(BASE / 'checkpoint')

def main():
    """Create the model and start the evaluation process."""
    LABELS=[]
    ignore_labels = []
    if args.classes == 'lip':
        ignore_labels = [0, 1, 2, 6, 9, 11, 16, 17]
    # grab the labels from user file
    try:
        with open(BASE / 'datasets' / 'labels' / '{}_labels.txt'.format(args.classes) , 'r') as f:
            for line in f:
                LABELS.append(line.strip('\n'))
    except Exception as e:
        print("{} No Label file for dataset {}".format(e,args.classes))
        sys.exit(1)

    make_dir_heirarchy(args.output_dir, LABELS)
    pattern_output_dir=''
    # PIPELINE: make output dir for pattern to classify after pipeline finished
    if args.pattern:
        pattern_output_dir = args.output_dir + '_pattern'
        make_dir_heirarchy(pattern_output_dir, LABELS)
        
    # buffer end is current index of last element in buffer when job ran
    # interval size is the size of the buffer jpp will categorize
    interval_size = int(args.interval_size)
    buffer_end = int(args.buffer_size)

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
    DATA_LIST = sorted_filename_list[max(0,buffer_end-interval_size) : buffer_end]
    
    # create label file for data loader and get number of IMAGES
    # image is concatenated with DATA_DIRECTORY in ImageReader init function!
    # filter videos (mainly for security when not using fullbody detection)
    NUM_STEPS = 0
    with open(DATA_LIST_PATH, 'w+') as f:
        for image in DATA_LIST:
            if image.split('.')[-1] in ['jpg','png','jpeg','JPG','PNG','JPEG']:
                f.write('/'+image+'\n')
                NUM_STEPS += 1
    if NUM_STEPS == 0:
        print("Exiting: No Images Found")
        return -1
    print("CLASSIFYING {} IMAGES".format(NUM_STEPS))

    #############################
    # LOAD NETWORK & DATA
    ############################
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False,
                             coord, buffer_end)
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
    ######################################################
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    print("Prediciting Now")
    # Iterate over training steps.
    masked_images = []
    for step in range(NUM_STEPS):
        parsing_ = sess.run(pred_all)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print(image_list[step])
        img_name = Path(image_list[step].split('/')[-1])
        img_path = os.path.join(DATA_DIRECTORY, img_name)
        if args.style_preprocess:
            msk = crop_images(img_path, parsing_, classes=args.classes)
        else:
            msk = segment_images(img_path, parsing_, classes=args.classes)
        if msk.size != 0:
            try:
                for msk_image, label in msk:
                    if label not in ignore_labels and valid_mask(msk_image):
                        masked_images.append([img_name, LABELS[label], msk_image])
            except Exception as e:
                print("JPP - ", e, img_name)
                continue
    coord.request_stop()
    coord.join(threads)
    if args.save_images:
        print("Saving Images")
        for name, label, image in masked_images:
            cv2.imwrite(str(Path(args.output_dir) / label / name), image)
            if args.pattern:
                cv2.imwrite(str(Path(pattern_output_dir) / label / name), image)
    
if __name__ == '__main__':
    main()


##############################################################333
