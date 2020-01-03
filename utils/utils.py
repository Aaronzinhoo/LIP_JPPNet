from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import os
import h5py
import scipy.misc
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

upper_clothing_index = 5
coat_index = 7
tops_index = [upper_clothing_index , coat_index]
pants_index = 9
skirt_index = 12
bottoms_index = [pants_index, skirt_index]
dress_index = 6
jumpsuit_index=10
full_index = [dress_index, jumpsuit_index]
n_classes = 20
# colour map
label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def valid_mask(mask , threshold=.01):
    """
    mask should be numpy array so operations can be performed by cv2
    returns True if percentage of colored pixels exceed threshold (keep this image)
    """
    MIN = np.array([1, 1, 1], np.uint8)
    MAX = np.array([255, 255, 255], np.uint8)
    dst = cv2.inRange(mask, MIN, MAX)
    return int(mask.size*threshold) < cv2.countNonZero(dst)

def make_dir_heirarchy(root_dir, subdirs):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    #create a blank outfile file for the mask
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i]))) #rows and columns of the image
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < n_classes:
                  pixels[k_,j_] = label_colours[k] #get the pixel at this point
      outputs[i] = np.array(img)
    return outputs

def _get_bbox_dim(image, all_contours=False):
    """
    returns cropped image that fits largest contour found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_mask = cv2.inRange(img_gray, 1, 255)
    contours, heirarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if all_contours:
        height, width = img_mask.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
        return max_y-min_y,max_x-min_x,min_y,min_x
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    return (h,w,y,x)

def segment_images(img_path, mask, classes='lip', num_images=1):
    """ Black out the original image where the pixels are categorized as unimportant objects"""

    conditions=[]
    indexes=[]
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    
    #create a copy of the original image
    image = Image.open(img_path)
    if len(image.split()) == 4 or image.mode == 'RGBA':
        image = image.convert('RGB')
    image_array = np.array(image) # should remove [] if you have more than one image
    image_array = image_array.reshape(1,image_array.shape[0],image_array.shape[1],
                                      image_array.shape[2])
    
    # conditions to mask the images clothing
    # this step create masks where false means the value isnt part of the class
    if classes=='fashion':
        top_mask_cond = [mask != upper_clothing_index, mask != coat_index]
        bottom_mask_cond = [mask != pants_index, mask != skirt_index]
        full_mask_cond = [mask != dress_index, mask != jumpsuit_index]
        conditions = [top_mask_cond, bottom_mask_cond, full_mask_cond]
        indexes = [tops_index, bottoms_index, full_index]
    if classes=='lip':
        for i in range(len(label_colours)):
            # skip hair and background labels
            if i==2 or i==0:
                continue
            conditions.append([mask != i])
            indexes.append([i])
        
    outputs = []
    #create a blank outfile file for the mask
    for i in range(num_images):
        # j will be class when returned to main
        for j, mask_cond in enumerate(conditions):
            # zero out objects that arent top/bottom/full
            clothes_idx = indexes[j]
            result = any(elem in mask  for elem in clothes_idx)
            if result:
                # get the mask from masked_where, returns true where mask condition is true
                # if the mask returns true for either index store it as this label 
                image_mask = np.ma.masked_where(np.all(mask_cond, axis=0), mask).mask
                image_copy = np.where(image_mask, label_colours[0], image_array)
                outputs.append((image_copy[0, :, :, :].astype('uint8'), j))
    return np.array(outputs)

def crop_images(img_path, mask, classes='lip', padding=100 ,num_images=1):
    """ Black out the original image where the pixels are categorized as unimportant objects"""

    conditions=[]
    indexes=[]
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    
    #create a copy of the original image
    image = cv2.imread(img_path)
    #image = Image.open(img_path)
    #image_array = np.array(image) # should remove [] if you have more than one image
    
    # conditions to mask the images clothing
    # this step create masks where false means the value isnt part of the class
    if classes=='fashion':
        top_mask_cond = [mask != upper_clothing_index, mask != coat_index]
        bottom_mask_cond = [mask != pants_index, mask != skirt_index]
        full_mask_cond = [mask != dress_index, mask != jumpsuit_index]
        conditions = [top_mask_cond, bottom_mask_cond, full_mask_cond]
        indexes = [tops_index, bottoms_index, full_index]
    if classes=='lip':
        for i in range(len(label_colours)):
            # skip hair and background labels
            if i==2 or i==0:
                continue
            conditions.append([mask != i])
            indexes.append([i])
    outputs = []
    # create a blank outfile file for the mask
    for i in range(num_images):
        # j will be class when returned to main
        for j, mask_cond in enumerate(conditions):
            # zero out objects that arent top/bottom/full
            clothes_idx = indexes[j]
            result = any(elem in mask for elem in clothes_idx)
            if result:
                # get the mask from masked_where, returns true where mask condition is true
                # if the mask returns true for either index store it as this label 
                image_mask = np.ma.masked_where(np.all(mask_cond, axis=0), mask).mask
                image_copy = np.where(image_mask, label_colours[0], image)
                h,w,y,x = _get_bbox_dim(image_copy[0].astype('uint8'),all_contours=True)
                y = max(0, y-padding)
                x = max(0, x-padding)
                image_copy = image[y:y+h+padding, x:x+w+padding]
                outputs.append((image_copy[:, :, :].astype('uint8'), j))
    return np.array(outputs)
    

def prepare_label(input_batch, new_size, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
          input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

def inv_preprocess(imgs, num_images):
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
  for i in range(num_images):
    outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
  return outputs


def store_many_hdf5(images, paths, label):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, X, X, 3) to be stored
        paths        paths array, (N, 1) to be stored
    """
    num_images = len(images)
    # Create a new HDF5 file
    file = h5py.File(f"{label}_pattern_images.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    paths_dataset = file.create_dataset(
        "paths", np.shape(paths), h5py.h5t.STD_U8BE, data=label
    )
    file.close()

def read_many_hdf5(label):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
 
        Returns:
        ----------
        images      images array, (N, X, X, 3) to be stored
        paths      associated meta data, str path (N, 1)
    """
    images,paths = [],[]
    # Open the HDF5 file
    file = h5py.File(f"{label}_pattern_images.h5", "r+")
    images = np.array(file["/images"]).astype("uint8")
    paths = np.array(file["/paths"]).astype("uint8")
    return zip(paths, images)

def save(saver, sess, logdir, step):
    '''Save weights.   
    Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
    '''
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
      
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        print("Restored model parameters from {}".format(ckpt_name))
        return True
    else:
        return False  
