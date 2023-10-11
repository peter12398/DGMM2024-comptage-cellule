# %%
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy
import keras.backend as K
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter
from tensorflow.keras import layers as kl
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from tensorflow.keras.callbacks import *
from tensorflow.python.keras import regularizers
from skimage.morphology import dilation, disk, square, opening, closing, label

from skimage.measure import block_reduce, regionprops
#sys.path.append("/home/xiaohu/PythonProject/DIMA_comtage_cellule/examples/segmentation_models_/models")
import morpholayers.layers as ml
import tensorflow as tf
import json
from skimage import io
from skimage.transform import resize
import sys
#sys.path.append("/home/xiaohu/PythonProject/DIMA_comtage_cellule/tensorblur")

import ipdb
print(tf.__version__)
print('It should be >= 2.0.0.')
"""
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
print(logical_devices)
"""

def count(images):
    """PLot images in one row."""
    tmp = np.sum(images)
    #print("sum = :", tmp)
    return tmp
    


@tf.function
def condition_equal(last,new,image):
    return tf.math.logical_not(tf.reduce_all(tf.math.equal(last, new)))


def update_dilation(last,new,mask):
     return [new, geodesic_dilation_step([new, mask]), mask]

@tf.function
def geodesic_dilation_step(X):
    """
    1 step of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation_step, name="reconstruction")([Mask,Image])
    """
    # perform a geodesic dilation with X[0] as marker, and X[1] as mask
    return tf.keras.layers.Minimum()([tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0]),X[1]])

@tf.function
def geodesic_dilation(X,steps=None):
    """
    Full reconstruction by dilation if steps=None, else
    K steps reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation, name="reconstruction")([Mask,Image])
    """
    rec = X[0]
    #Full reconstruction is steps==None by dilation, else: partial reconstruction
    rec = geodesic_dilation_step([rec, X[1]])
    _, rec,_=tf.while_loop(condition_equal, 
                            update_dilation, 
                            [X[0], rec, X[1]],
                            maximum_iterations=steps)
    return rec

def reconstruction_dilation(X):
    """
    Full geodesic reconstruction by dilation, reaching idempotence
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    return geodesic_dilation(X, steps=None)


# In[3]:


@tf.custom_gradient
def tfround(x):
    def grad(dy):
        return dy
    return tf.round(x), grad


# In[4]:


class Sampling(tf.keras.layers.Layer):
    """Sampling Random Uniform."""

    def call(self, inputs):
        dim = tf.shape(inputs)
        epsilon = tf.keras.backend.random_uniform(shape=(dim))
        return epsilon



# helper function for data visualization
def visualize(ids,**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    if not os.path.exists(ROT_PATH+"/visualize_test_{}".format(exp_name)):
        os.makedirs(ROT_PATH+"/visualize_test_{}".format(exp_name))
    plt.savefig(ROT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,ids))


import morpholayers.layers as ml
from tensorflow.keras import layers as kl
from skimage.morphology import area_opening
from skimage.morphology import label
input_shape = [256,256,1] 
  
class StepsGeodesicDilationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(StepsGeodesicDilationLayer, self).__init__()
    def call(self, inputs):
        return ml.geodesic_dilation(inputs,steps=5)
#Model to Compute Geodesic Reconstruction in steps.

xin=kl.Input(shape=input_shape)
xMask=kl.Input(shape=input_shape)
#xout=kl.Lambda(ml.geodesic_dilation, name="reconstruction")([xMask,xin])
xout=StepsGeodesicDilationLayer()([xMask,xin])
#xout=ml.geodesic_dilation([xMask,xin])
modelREC_=tf.keras.Model(inputs=[xin,xMask],outputs=xout)
modelREC_.summary()
modelREC_.compile()

"""   
# helper function for data visualization
def visualize(**images):
    #PLot images in one row.
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig("visualize_{}.jpg".format(exp_name))
"""
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
    
NORMALISE01 = False

if not NORMALISE01:
    dir_name = "best_h_dataset255"
    #dir_name = "best_h_dataset255_new"
else:
    dir_name = "best_h_dataset01"
print("dir:{} used".format(dir_name))    

ROOT_PATH = "/home/xiaohu/workspace/MINES/DGMM2024_comptage_cellule"
output_npy_save_path = ROOT_PATH + "/{}/ouput_np".format(dir_name)
output_h_file_save_path = ROOT_PATH + "/{}/best_h".format(dir_name)
input_npy_save_path = ROOT_PATH + "/{}/input_np".format(dir_name)

# classes for data loading and preprocessing
class Dataset:
    """
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            ids = None,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
        
    ):  
        if ids:
            self.ids = ids
        else:
            self.ids = os.listdir(images_dir)
        print("len self.ids={}".format(len(self.ids)))
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.h_th_dict = {}
        self.SET = images_dir.split("/")[-2]

        with open(output_h_file_save_path + '/best_h_opening_closing_{}.json'.format(self.SET), 'r') as f:
            self.h_th_dict = json.load(f)

        
        # convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        #self.augmentation = augmentation
        #self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #print(self.images_fps[i])
        image_name = self.images_fps[i].split("/")[-1].split(".")[0]
        set_name = self.images_fps[i].split("/")[-3]
        #image = cv2.imread(self.images_fps[i])[:,:,1:2]
        
        #ipdb.set_trace()
        imColLarge = io.imread(self.images_fps[i])
        h,w = imColLarge.shape[0], imColLarge.shape[1]
        imCol = resize(imColLarge,(h//4, w//4))
        if not NORMALISE01:
            imCol = (255*imCol).astype('uint8')
        else:
            imCol = (imCol[:,:,1])
        
        input_np_path = input_npy_save_path + "/" + set_name 
        image = np.load(input_np_path + "/" + image_name + "_after_opening_closing.npy")[0,:,:,0]
        #image = imCol[:,:,1]
        #image = np.repeat(image, 3, axis=-1)
        
    
        #gt_h = np.array(float(self.h_th_dict[str(image_name)]['h']))
        gt_h = np.array(float(self.h_th_dict[str(image_name)]))

        #gt_threshold = np.array(float(self.h_th_dict[image_name]['threshold']))
        
        #h,w = image.shape[0], image.shape[1]
        #image = block_reduce(image, block_size=(4,4,1), func=np.max)
        #image = cv2.resize(image,(h//4, w//4))/255.
        #print("image max:",image.max())
        #print("image sum:",np.sum(image))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #mask = cv2.imread(self.masks_fps[i], 0)
        mask = io.imread(self.masks_fps[i])
        imLabels = block_reduce(mask, block_size=(4,4), func=np.max)
        ntruth = int(imLabels.sum()/255)
        #print("mask:",mask.shape)
        #print("mask max:",mask.max())
        #print("mask sum:",np.sum(mask))
        #mask = np.array(block_reduce(mask, block_size=(4,4), func=np.max),dtype=np.uint8)
        mask = block_reduce(mask, block_size=(4,4), func=np.max)
        #mask = mask/255.0
        ##print("mask:",mask.shape)
        #print("mask max:",mask.max())
        #print("mask sum:",np.sum(mask))
    
        #mask = gaussian_filter(mask, sigma=4)*200
        
        #mask = gaussian_filter(mask, sigma=1)
        #mask = mask*(image)
        #mask = mask/np.max(mask)
        
        #REC=modelREC_.predict([np.expand_dims(np.expand_dims(image,axis=0),axis=-1),np.expand_dims(np.expand_dims(mask,axis=0),axis=-1)],verbose=0)
        npy_path_to_load = output_npy_save_path + "/" +set_name + "/" + image_name + ".npy"
        #print("npy loaded from:", npy_path_to_load)
        rec_ = np.load(npy_path_to_load)
        
        npy_path_to_load = output_npy_save_path + "/" +set_name + "/" + image_name + "_imMax_best.npy"
        #print("npy loaded from:", npy_path_to_load)
        imMax_best = np.load(npy_path_to_load)/255.
     
        if mask.max()>0:
            mask = mask / mask.max()
        #mask = np.array(scipy.ndimage.morphology.binary_dilation(mask, iterations = 8),dtype=np.uint8)
        
        """
        num_class = 2
        #mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        #new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
        """
        #print("mask:",mask.shape)
        #print("mask max:",mask.max())
        #print("mask sum:",np.sum(mask))
        #mask = cv2.imread(self.masks_fps[i],cv2.IMREAD_GRAYSCALE)
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        """
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        """   
        #return image[::2,::2], mask[::2,::2]
        #if self.SET == 'set2':
        #return image, mask, REC[0,:,:,0], image_name ,gt_h , gt_threshold
        return image, mask, rec_, image_name ,gt_h, ntruth, imMax_best, imCol #, gt_threshold
    
        #elif self.SET == 'set1':
        #return image, REC[0,:,:,0]
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, train=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.train = train

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            #data.append(tuple((self.dataset[j][0],self.dataset[j][2])))
            data.append(list((self.dataset[j][0],self.dataset[j][2], self.dataset[j][4], self.dataset[j][5], self.dataset[j][6])))
            #data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        #batch = [batch[0], list((batch[1],batch[2]))]
        
        if self.train:
            
            if int(np.random.randint(2, size=1) % 2):
                #print("image and rec flip_up_down used")
                #batch[0] = tf.image.flip_up_down(np.expand_dims(batch[0],axis=-1))[:,:,:,0].numpy()
                #batch[1] = tf.image.flip_up_down(np.expand_dims(batch[1],axis=-1))[:,:,:,0].numpy()
                
                batch[0] = np.flip(batch[0], 2)
                batch[1] = np.flip(batch[1], 2)
                batch[-1] = np.flip(batch[-1], 2)
                
                
            if int(np.random.randint(2, size=1) % 2):
                #print("image and rec flip_left_right used")
                #batch[0] = tf.image.flip_left_right(np.expand_dims(batch[0],axis=-1))[:,:,:,0].numpy()
                #batch[1] = tf.image.flip_left_right(np.expand_dims(batch[1],axis=-1))[:,:,:,0].numpy()
                
                batch[0] = np.flip(batch[0], 1)
                batch[1] = np.flip(batch[1], 1)
                batch[-1] = np.flip(batch[-1], 1)
            
            """"
            if int(np.random.randint(2, size=1) % 2):
                x_roll_dis =  np.random.randint(1,batch[0].shape[1])
                batch[0] = np.roll(batch[0],x_roll_dis, axis=1)
                batch[1] = np.roll(batch[1],x_roll_dis, axis=1)
                
            if int(np.random.randint(2, size=1) % 2):
                y_roll_dis =  np.random.randint(1,batch[0].shape[2])
                batch[0] = np.roll(batch[0],y_roll_dis, axis=2)
                batch[1] = np.roll(batch[1],y_roll_dis, axis=2)
            """
                
            if int(np.random.randint(2, size=1) % 2):
                rotate_num =  np.random.randint(1,4)
                batch[0] = np.rot90(batch[0], rotate_num, (1,2))
                batch[1] = np.rot90(batch[1], rotate_num, (1,2))
                batch[-1] = np.rot90(batch[-1], rotate_num, (1,2))
        else:
            pass


        #batch = [batch[0], batch[1]]
        batch[1] = np.expand_dims(batch[1],axis=-1)
        batch[-1] = np.expand_dims(batch[-1],axis=-1)
        #batch[2] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[2],axis=-1),axis=-1),axis=-1)
        #batch[3] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[3],axis=-1),axis=-1),axis=-1)
        batch = [batch[0], list((batch[1],batch[2],batch[3],batch[-1]))]
        #batch = [batch[0], batch[3]]
        
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   


# %%




class Hextrema(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Hextrema, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Hextrema, self).build(input_shape)

    def call(self, x):
        return ml.h_maxima_transform([x[0], x[1]])

class h_extrema_denoising_block(tf.keras.Model):
    def __init__(self, dropout=0.1, name="mse_denoising"):       
        super(h_extrema_denoising_block, self).__init__(name=name)
        """
        self.conv1 = kl.Conv2D(16, kernel_size=(3,3), 
                                padding="valid",kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5))
        self.conv2 = kl.Conv2D(16, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5))
        self.conv3 = kl.Conv2D(32, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5))
        self.conv4 = kl.Conv2D(32, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5))
        """
        
        self.conv1 = kl.Conv2D(16, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',)
        self.conv2 = kl.Conv2D(16, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',)
        self.conv3 = kl.Conv2D(32, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',)
        self.conv4 = kl.Conv2D(32, kernel_size=(3,3), padding="valid",kernel_initializer='glorot_uniform',)
        self.maxpooling = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.dropout = kl.Dropout(dropout)
        #self.globalavgpooling = kl.GlobalAveragePooling2D(data_format='channels_last')
        self.globalavgpooling = kl.GlobalMaxPooling2D(data_format='channels_last')
        
        #self.dense = kl.Dense(1,kernel_constraint=tf.keras.constraints.NonNeg(),name="h_denoising")
        #self.dense = kl.Dense(1,kernel_initializer='zeros',bias_initializer=tf.keras.initializers.Constant(value=.1),name="h_denoising")
        self.dense = kl.Dense(1,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None),bias_initializer=tf.keras.initializers.Constant(value=.1),name="h_denoising")
        
        self.h_extrema_transform = Hextrema()
        self.batchnorm1 = kl.BatchNormalization()
        self.batchnorm2 = kl.BatchNormalization()
        self.batchnorm3 = kl.BatchNormalization()
        self.batchnorm4 = kl.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.batchnorm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = tf.nn.relu(x)
        #x = self.maxpooling(x)
        x = self.globalavgpooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        #x = tf.nn.sigmoid(x)
        h = tf.expand_dims(tf.expand_dims(x,axis=-1),axis=-1)
        return h

class h_extrema_denoising_block2(tf.keras.Model):
    def __init__(self, dropout=0.2, name="h_extrema_denoising_block2"):       
        super(h_extrema_denoising_block2, self).__init__(name=name)
        self.conv1 = kl.Conv2D(filters = 8, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv2 = kl.Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv3 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv4 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.maxpooling = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.maxpooling2 = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        #self.maxpooling3 = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        #self.maxpooling4 = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        
        self.dropout = kl.Dropout(dropout)
        #self.dropout2 = kl.Dropout(dropout)
        #self.dropout3 = kl.Dropout(dropout)
        #self.dropout4 = kl.Dropout(0.4)
        #self.globalavgpooling = kl.GlobalAveragePooling2D(data_format='channels_last')
        self.globalavgpooling = kl.GlobalMaxPooling2D(data_format='channels_last')
        #self.dense = kl.Dense(1,kernel_constraint=tf.keras.constraints.NonNeg(),name="h_denoising")
        #self.dense = kl.Dense(64, activation='relu')
        #self.dense2 = kl.Dense(128, activation='relu')
        self.dense = kl.Dense(1,kernel_constraint=tf.keras.constraints.NonNeg(),name="h_denoising")
        #self.h_extrema_transform = Hextrema()
        #self.flatten = kl.Flatten()
        self.batchnorm1 = kl.BatchNormalization()
        self.batchnorm2 = kl.BatchNormalization()
        self.batchnorm3 = kl.BatchNormalization()
        self.batchnorm4 = kl.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        #x = tf.nn.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        #x = tf.nn.relu(x)
        x = self.maxpooling(x)
        
        
        x = self.batchnorm2(x)
        
        #x = self.maxpooling2(x)
        #x = self.dropout(x)
        #x = tf.nn.relu(x)
        x = self.conv3(x)
        #x = tf.nn.relu(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        #x = tf.nn.relu(x)
        x = self.maxpooling2(x)
        
        
        #x = self.batchnorm4(x)
        #x = tf.nn.relu(x)
        #x = self.maxpooling4(x)
        
        x = self.globalavgpooling(x)
        #x = self.dropout(x)
        #x = self.flatten(x)
        
        #x = self.dense(x)
        #x = tf.nn.relu(x)
        #x = self.dropout3(x)
        
        #x = self.dense2(x)
        #x = tf.nn.relu(x)
        #x = self.batchnorm4(x)
        #x = self.dropout(x)
        
        x = self.dense(x)
        h = tf.expand_dims(tf.expand_dims(x,axis=-1),axis=-1)
        return h

class Hextrema_layer_(tf.keras.layers.Layer):
    def __init__(self, ):
        super(Hextrema_layer_, self).__init__()
        self.h_extrema_denoising_block_ = h_extrema_denoising_block2()
        self.Hextrema = Hextrema(name="mse_denoising")
        #self._name = "Hextrema_layer_"
        self.subtracte_layer = kl.Subtract()
        sz = 3
        b = square(sz) # structuring element
        
        self.structure_element = tf.expand_dims(tf.convert_to_tensor(b, dtype=tf.float32), axis=-1)#在第一维增加一维
        self.opening_2D_layer = ml.Opening2D(num_filters = 1, kernel_size = (3,3))
        self.closing_2D_layer = ml.Closing2D(num_filters = 1, kernel_size = (3,3))

        #self.opening_layer = ml.Opening2D(num_filters=1, kernel_size=(3,3))
        
    def build(self, input_shape):
        super(Hextrema_layer_, self).build(input_shape)

    def call(self, x):
        
        xin = x #kl.Input(shape=input_shape)
        x = self.opening_2D_layer(xin)
        x = self.closing_2D_layer(x)  #ml.closing2d(x, b, strides = (1,1), padding = "same")
            
        #xrec=kl.Lambda(ml.geodesic_dilation, name="reconstruction")([x-h,x])
        
        #x = ml.opening2d(x, self.structure_element, strides = (1,1), padding = "same")
        #x = ml.closing2d(x, self.structure_element, strides = (1,1), padding = "same")

        h = self.h_extrema_denoising_block_(x)
        output_h_denoising = self.Hextrema([h, x])
        
        #rmax_reg = kl.Lambda(ml.region_maxima_transform)(output_h_denoising)
        #x = kl.Multiply(name="r_max_reg")([output_h_denoising, 255.*rmax_reg])
        #out =  kl.Lambda(lambda x: ((1./ (1. + tf.exp(- 1e5*x))) - 0.5)*2 )(x)
        out = output_h_denoising #tf.keras.activations.sigmoid(x) 
        
        """
        output_h_denoising = self.Hextrema([h, x])
        print("Hextrema_layer_ called")
        #out = self.subtracte_layer([x,output_h_denoising])
        #print("substraction called")
        rmax_reg = kl.Lambda(ml.region_maxima_transform)(output_h_denoising)
        #rmax_reg = kl.Multiply(name="r_max_reg")([output_h_denoising, rmax_reg])
        out = kl.Multiply(name="r_max_reg")([output_h_denoising, 255.*rmax_reg])
        #rmax_reg = self.opening_layer(rmax_reg)
        #rmax_reg = kl.Multiply(name="r_max_reg")([output_h_denoising, rmax_reg])
        #rmax_reg = kl.MaxPool2D(pool_size=(2,2), padding="valid",name="r_max_reg")(rmax_reg)
        # Models Creation
        """
        #output = self.layer(output_h_denoising)
        return out,h #output_h_denoising, rmax_reg, h
 
def detectCells(im, NORMALISE01 = False):
    #imOpen = opening(im, b)
    #imAF = closing(imOpen, b)
    #imMax, imRec  = hMax(imAF, h, disk(1))

    xrec=kl.Input(shape=input_shape)
  
    if NORMALISE01:
        delta = tf.constant(1/255., dtype=tf.float32)
    else:
        delta = tf.constant(1., dtype=tf.float32)
    xrec2=kl.Lambda(ml.geodesic_dilation, name="reconstruction2")([xrec-delta,xrec])
    

    #rmax_reg = kl.Lambda(ml.region_maxima_transform)(xrec)
    #xout = kl.Multiply(name="r_max_reg")([xrec, 255.*rmax_reg])
    
    #imMax = tf.zeros_like(xout, dtype = tf.float32)
    #idxs = tf.where(xout > imMax)
    #imMax[idxs] = 1.
    
    modelREC=tf.keras.Model(inputs=xrec,outputs=[xrec,xrec2])
    modelREC.compile()
    
    xrec,xrec2 = modelREC.predict(im,verbose=0)
    xrec2 = np.squeeze(xrec2)
    xrec = np.squeeze(xrec)
    #imMax,imRec = modelREC.predict(np.expand_dims(np.expand_dims(im,axis=0),axis=-1),verbose=0)
    
    imMax = np.zeros(xrec2.shape)
    idxs = np.where(xrec2 < xrec)
    imMax[idxs] = 255
    
    cc, ncells = label(imMax, return_num = True, connectivity = 2)
    props = regionprops(cc)

    imDetec = np.zeros(imMax.shape)
    for k in range(len(props)):
        xc = int(props[k].centroid[0])
        yc = int(props[k].centroid[1])
        imDetec[xc, yc] = 255
    imDetec = dilation(imDetec, square(3))
    return ncells, imDetec, xrec



class H_maxima_model:
    def __init__(self, input_shape, train_dataloader, valid_dataloader, test_dataloader, test_dataset, EPOCHS=100, BATCH_SIZE = 16, loss = 'binary_crossentropy', loss_weights = None , lr = 0.01,  IMAGE_SAVE_PATH = "/home/xiaohu/PythonProject/DIMA_comtage_cellule/visualize_main", MODE = "post_processor", metrics = None, resume = False) -> None:
        self.input_shape = input_shape
        self.MODE = MODE
        print("MODE = {}".format(self.MODE))
        
        self.h_extrema_denoising_block_ = h_extrema_denoising_block2()
        self.Hextrema = Hextrema()

        #self.opening_2D_layer = ml.Opening2D(num_filters = 1, kernel_size = (3,3))
        #self.closing_2D_layer = ml.Closing2D(num_filters = 1, kernel_size = (3,3))
        
        self.nn, self.nn_h, self.nn_n, self.nn_binaraliszd = self.get_simple_model(self.input_shape)
        self.nn_epochs = EPOCHS
        self.nn_batch_size = EPOCHS
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.metrics = metrics
        #self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics = self.metrics)
        self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=None), metrics = self.metrics)
        self.resume = resume
        
        if self.resume:
            best_weight_load_path = ROT_PATH + '/best_model_{}.h5'.format(exp_name)
            self.nn.load_weights(best_weight_load_path)
            print("load weight from :{}".format(best_weight_load_path))
        
        #self.nn.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr), metrics = self.metrics)
        print("lr =",self.lr)
        print("metrics =",self.metrics)
        print("self.nn.compile done\n")
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.test_dataset = test_dataset
        self.IMAGE_SAVE_PATH = IMAGE_SAVE_PATH
        


        
    def get_simple_model(self, input_shape):
        xin=kl.Input(shape=input_shape)
        
        #Hextrema_layer = Hextrema_layer_()
            
        #x = self.opening_2D_layer(xin)
        #x = self.closing_2D_layer(x)  #ml.closing2d(x, b, strides = (1,1), padding = "same")
        x = xin   
        #xrec=kl.Lambda(ml.geodesic_dilation, name="reconstruction")([x-h,x])
        
        #x = ml.opening2d(x, self.structure_element, strides = (1,1), padding = "same")
        #x = ml.closing2d(x, self.structure_element, strides = (1,1), padding = "same")

        h = self.h_extrema_denoising_block_(x)
        h_ret = tf.math.reduce_sum(h,axis=[1,2,3])
        
        rec = self.Hextrema([h, x])
        
        delta = tf.constant(1., dtype=tf.float32)
        rec2 = kl.Lambda(ml.geodesic_dilation, name="reconstruction2")([rec-delta,rec])
        
        size_im = self.input_shape #(256,256,1)
        c=10*size_im[0]*size_im[1]*size_im[2] #Different should be greater that one to be detected in the approximation of =
        InLayer = rec-rec2
        
        
        U=tf.ones_like(InLayer)
        M=tf.keras.layers.Minimum()([U,InLayer*c])
        imMax=tf.nn.relu((1-tf.math.abs((U-M))*c)) #Approximation of =
        
        #lambda_ = tf.constant(1e-4, dtype=tf.float32)
        #imMax = 1 - tf.math.exp(-(InLayer*255.)**2/lambda_) 
        #imMax = InLayer
        
        #imMax = tf.nn.sigmoid(InLayer)
        
        #model_binarilize=tf.keras.Model(InLayer,imMax) 
        #imMax = model_binarilize(np.expand_dims(np.expand_dims((xrec-xrec2),axis=0),axis=-1)).numpy().squeeze()
        InLayer = imMax
        U=Sampling()(InLayer)
        U=U*c
        M=tf.keras.layers.Minimum()([U,InLayer*c])
        R= tf.keras.layers.Lambda(geodesic_dilation,name="rec")([M,InLayer*c])
        #NCC=tf.math.equal(U,R) #This is not differentiable.
        Detection=tf.nn.relu(1-tf.math.abs(U-R)) #Approximation of =
        NCC=tf.math.reduce_sum(Detection,axis=[1,2,3])
        #model=tf.keras.Model(InLayer,Detection) #Only for interpretation
        #modeltoCount=tf.keras.Model(InLayer,NCC) #Model to Count

        #K = Detection #model(np.expand_dims(np.expand_dims(imMax,axis=0),axis=-1))
        #cetroids_diff = K[0] #.numpy()
        #cetroids_diff = dilation(cetroids_diff.squeeze(), square(3))
        detected_diff = NCC #modeltoCount(np.expand_dims(np.expand_dims(imMax,axis=0),axis=-1)).numpy()[0]

        return tf.keras.Model(xin,[rec,h_ret,detected_diff,imMax]), tf.keras.Model(xin,h), tf.keras.Model(xin,detected_diff), tf.keras.Model(xin,imMax)
        #return tf.keras.Model(xin,final), tf.keras.Model(xin,xout_tmp), tf.keras.Model(xin,h)
    
    def predict_output(self, X):
        return self.nn.predict(X)
    
    def predict_h(self, X):
        return self.nn_h.predict(X)
    
    def predict_n(self, X):
        return self.nn_n.predict(X)
    
    def predict_binaraliszd(self, X):
        return self.nn_binaraliszd(X)

    def train(self, verbose=1):
        if verbose == 1:
            self.nn.summary()

        #Callback definition
        CBs = [
            tf.keras.callbacks.ModelCheckpoint(ROT_PATH + '/best_model_{}.h5'.format(exp_name), monitor='val_loss', verbose=1 ,save_weights_only=True, save_best_only=True, mode='min', period=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=0.0001),
            tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=300, verbose=1, mode='auto',
                              baseline=None, restore_best_weights=False)
            #ModelCheckpoint('best_model.h5', monitor='loss',verbose=1, save_best_only=True)
        ]
        #Training the model
        self.history = self.nn.fit(
                self.train_dataloader, 
                steps_per_epoch=len(self.train_dataloader), 
                epochs=self.nn_epochs, 
                validation_data=self.valid_dataloader, 
                validation_steps=len(self.valid_dataloader),
                callbacks=CBs,
            )
        
        plt.figure()
        plt.plot(self.history.history['loss'],label='loss')
        plt.plot(self.history.history['val_loss'],label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.IMAGE_SAVE_PATH+"/loss.jpg")
        plt.close()
        
        
    def test(self,):      
        #scores = self.nn.evaluate_generator(self.test_dataloader)
        #metrics =['mean_absolute_error']
        #print("Loss: {:.5}".format(scores))
        #for metric, value in zip(metrics, scores[1:]):
        #    print("mean {}: {:.5}".format(metric, value))
            
        #n = 46
        best_weight_load_path = ROT_PATH + '/pretrained_model_weight/best_model_{}.h5'.format(exp_name)
        self.nn.load_weights(best_weight_load_path)
        print("load weight from :{}".format(best_weight_load_path))
        
        ids = np.arange(len(self.test_dataset)) #np.random.choice(np.arange(len(test_dataset)), size=n)

        rat_err_list = []
        rat_err_diff_list = []
        rat_err_nomerator_list = []
        
        count_err_nomerator_list = []
        count_err_diff_list = []
        average_relative_error_list = []
        
        pr_h_list = []
        gt_h_list = []
        
        n_detec_list = []
        n_gt_list = []
        for i in ids:
            
            #image, gt_mask, rec_, image_name ,gt_h , gt_threshold = self.test_dataset[i]
            image, gt_mask, rec_, image_name ,gt_h, ntruth, imMax_best, imCol  = self.test_dataset[i]
            image_expanded = np.expand_dims(image, axis=0)
            pr_mask = self.predict_output(image_expanded)[0] #self.predict_output(image_expanded) #self.predict_output(image_expanded)[0]
            
            pr_h =  self.predict_output(image_expanded)[1].item() #self.predict_h(image_expanded) #self.predict_output(image_expanded)[1].item() #self.predict_h(image)
            pr_h_list.append(pr_h)
            gt_h_list.append(gt_h)
            
            pr_n =  self.predict_n(image_expanded).item()
            pr_binarized_map =  self.predict_binaraliszd(image_expanded).numpy().squeeze()
            
            imTruth = np.zeros((256,256,3),dtype=np.uint8)
            image = imCol[:,:,1]
            #imRGB = np.zeros((256,256,3))
            #imLabels = dilation(gt_mask, square(3))
            imLabels = gt_mask
            imLabels = dilation(imLabels, square(3))
            idxs = np.where(imLabels > 0)
            for k in range(3):
                #if not NORMALISE01:
                #    imTruth[:,:,k] = image/255 #im #im/255
                #else:
                imTruth[:,:,k] = image #im/255
                if k == 0:
                    imTruth[idxs[0], idxs[1], k] = 255
                else :
                    imTruth[idxs[0], idxs[1], k] = 0
            
            ndetec, imDetec, imRec_best = detectCells(pr_mask, NORMALISE01)
            print("For h = "+ str(pr_h) + ", truth: " + str(ntruth) + " ; Detected: " + str(ndetec))
            
            n_detec_list.append(ndetec)
            n_gt_list.append(ntruth)
            
            count_err_diff_list.append(np.abs(ntruth - ndetec))
            count_err_nomerator_list.append(np.abs(ntruth))
            if np.abs(ntruth)!= 0:
                average_relative_error_list.append(np.abs(ntruth - ndetec)/np.abs(ntruth))
            
            imDetecCol = np.zeros((256,256,3),dtype=np.uint8)
            idxs = np.where(imDetec > 0)
            for k in range(3):
                #if not NORMALISE01:
                #    imDetecCol[:,:,k] = image/255
                #else:
                imDetecCol[:,:,k] = image #im/255
                if k == 0:
                    imDetecCol[idxs[0], idxs[1], k] = 255
                else :
                    imDetecCol[idxs[0], idxs[1], k] = 0

            #ipdb.set_trace()
            """
            plt.figure()
            plt.imshow(imCol)
            plt.title('Input RGB image (resized)')
            plt.axis('off')

            plt.savefig(ROT_PATH+"/visualize_test_{}/{}_Input.jpg".format(exp_name,i))
            
            plt.figure()
            plt.title('Green channel and ground truth ('+str(ntruth)+' cells).')
            plt.imshow(imTruth)
            plt.axis('off')

            plt.savefig(ROT_PATH+"/visualize_test_{}/{}_imTruth.jpg".format(exp_name,i))
            
            plt.figure()
            plt.title('Detections for h = {}'.format( '%.2f'%(float(pr_h)) ) + ' ('+ str(ndetec)+' cells).')
            plt.axis('off')

            plt.imshow(imDetecCol)
            plt.savefig(ROT_PATH+"/visualize_test_{}/{}_imDetecCol.jpg".format(exp_name,i))
            """
            
            
            """
            plt.figure()
            plt.title('pr_binarized_map for h = {}'.format('%.2f'%float(pr_h)))
            plt.imshow(pr_binarized_map)
            """
            #ipdb.set_trace()
            save_path = ROT_PATH+"/visualize_test_{}".format(exp_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(save_path+"/{}_Input.png".format(i), imCol)
            plt.imsave(save_path+"/{}_imTruth_{}.png".format(i,ntruth), imTruth)
            plt.imsave(save_path+"/{}_imDetecCol_{}.png".format(i,ndetec), imDetecCol)
            
            """
            tmp = pr_mask[0,:,:,1].squeeze()
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            pr_mask=np.array(tmp, dtype=np.float64)
            
            print("gt_mask min",gt_mask.min())
            print("gt_mask max",gt_mask.max())
            print("pr_mask min",pr_mask.min())
            print("pr_mask max",pr_mask.max())
            
            
            visualize(
                ids = i,
                image=denormalize(image.squeeze()),
                #gt_mask=gt_mask[:,:,1].squeeze(),
                #pr_mask=pr_mask,
                pr_mask=pr_mask.squeeze(),
                rec_ = rec_.squeeze(),
                #pr_mask=pr_mask.squeeze(),
                #pr_interm = pr_interm.squeeze()
            )"""
            
            if count(gt_mask) < 0.001:
                continue
            else:
                print("count(pr_h):",count(pr_h))
                print("count(gt_h):",count(gt_h))
                rat_err_diff_list.append(np.abs(count(pr_h)- count(gt_h)))
                rat_err_nomerator_list.append( (count(gt_h)))
                if (count(gt_h)) > 1e-3:
                    rat_err = np.abs((count(pr_h)- count(gt_h)) ) / (count(gt_h))
                    rat_err_list.append(rat_err)
            
            #rat_err = np.abs(count(pr_mask)- count(gt_mask)*100.0 ) / (count(gt_mask)*100.0)
            #print("rat_err =:", rat_err)

        print("len(pr_h_list):", len(pr_h_list))
        #ipdb.set_trace()
        plt.close()
        plt.figure()
        pr_h_array = np.array(pr_h_list)
        gt_h_array = np.array(gt_h_list)
        plt.plot(pr_h_array, label='pr_h_array')
        plt.plot(gt_h_array, label='gt_h_array')
        plt.legend()
        plt.title("test set predited h")
        plt.savefig(ROT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,'pr_h_array'))
        

        n_detec_array = np.array(n_detec_list)
        n_gt_array = np.array(n_gt_list)
        
        fig = plt.figure()
        
        min_n = np.min([np.min(n_gt_array),np.min(n_detec_array)])
        max_n = np.max([np.max(n_gt_array),np.max(n_detec_array)])
        
        ax = fig.add_subplot(111)
        plt.plot(n_gt_array, n_detec_array, 'o')
        plt.plot(np.linspace(min_n,max_n,100), np.linspace(min_n,max_n,100))
        plt.xlabel('True cell number')
        plt.ylabel('Estimated cell number')
        
        for i in range(len(n_gt_array)):                           
            ax.annotate('%s' % str(i), xy = [n_gt_array[i], n_detec_array[i]] , textcoords='data')
        plt.title('Predicted vs true cell number, train set')
        plt.savefig(ROT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,'n_detect_and_n_gt'))
        plt.show()

        
        fig = plt.figure()
        
        min_h = np.min([np.min(gt_h_array),np.min(pr_h_array)])
        max_h = np.max([np.max(gt_h_array),np.max(pr_h_array)])
        
        ax = fig.add_subplot(111)
        plt.plot(gt_h_array, pr_h_array, 'o')
        plt.plot(np.linspace(min_h,max_h,100), np.linspace(min_h,max_h,100))
        plt.xlabel('True h')
        plt.ylabel('Estimated h')
        
        for i in range(len(gt_h_array)):                           
            ax.annotate('%s' % str(i), xy = [gt_h_array[i], pr_h_array[i]] , textcoords='data')
        plt.title('Predicted vs true h, train set')
        plt.savefig(ROT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,'pr_h_and_gt_h'))
        plt.show()
                
        #print("rat_err_avg =:", np.sum(rat_err_diff_list)/ np.sum(rat_err_nomerator_list))
        print("rat_err_avg =:", np.average(rat_err_list))
        print("average relative error =:", np.average(average_relative_error_list))
        print("total relative error =:", np.sum(count_err_diff_list)/ np.sum(count_err_nomerator_list))

                


"""

N = 36
image, mask = dataset[N] 
Y_pred=model.predict(np.expand_dims(image,axis=0))
plt.figure(figsize=(24,24))
plt.subplot(1,3,3)
plt.imshow(Y_pred[0])
plt.axis('off')
plt.subplot(1,3,1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(mask)
plt.axis('off')
plt.savefig(IMAGE_SAVE_PATH+"/{}.jpg".format(N))
plt.show()
"""

# %%

if __name__ == "__main__":
    import ipdb
    
    IMAGE_SAVE_PATH = ROOT_PATH + "/visualize_main"
    ROT_PATH = ROOT_PATH #"/home/xiaohu/PythonProject/DIMA_comtage_cellule"
    DATA_DIR = '/home/xiaohu/workspace/MINES/segmentation_models_400epochs_copy/examples/data_/database_melanocytes_trp1/'
    #DATA_DIR = '/Users/santiago1/Downloads/database_melanocytes_trp1/'
    # load repo with data if it is not exists

    random_seed = 4
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    x_train_dir = os.path.join(DATA_DIR, 'set1', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'set1', 'labels')

    x_test_dir = os.path.join(DATA_DIR, 'set2', 'images')
    y_test_dir = os.path.join(DATA_DIR, 'set2', 'labels')

    x_valid_dir = x_train_dir #os.path.join(DATA_DIR, 'val')
    y_valid_dir = y_train_dir #os.path.join(DATA_DIR, 'valannot')

    exp_name = "only_hmaxima" #"1hmax_layer_out_conv_sigmoid_jaccardloss" #"1hmax_layer_out_conv_sigmpoid_l1loss_100gaussian"
                    
    # Lets look at data we have
    dataset = Dataset(x_train_dir, y_train_dir)




    # %%
    #!git clone https://github.com/Jacobiano/morpholayers.git

    # %%
    BATCH_SIZE=16
    
    ids_images_train = os.listdir(x_train_dir)
    TRAIN_VAL_SPLIT = 0.8
    len_train = len(ids_images_train)
    idx_val = int(TRAIN_VAL_SPLIT*len_train)
    
    
    np.random.shuffle(ids_images_train)
    # Reserve 10,000 samples for validation
    ids_images_val = ids_images_train[idx_val:]
    ids_images_train = ids_images_train[:idx_val]
    print("train len:{}; val len:{}".format(len(ids_images_train), len(ids_images_val)))
    
    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        ids = ids_images_train,
    )
    # Dataset for validation images
    
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        ids = ids_images_val,
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, train = True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False, train = False)

    
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
    )
    
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train = False)

    # %%
    input_shape = [256,256,1] 
    EPOCHS = 1600
    LR = .001
    BATCH_SIZE = 16
    NORMALISE01_ = False
    assert NORMALISE01_ == NORMALISE01
    RESUME = False
    
    lambda_h = 0.1 # hyperparameter to be adjusted
    def joint_loss(y_true, y_pred):
        # 
        print("y_true:",y_true)
        print("y_pred:",y_pred)
        rec_loss = K.mean(K.square(y_true[0] - y_pred[0]))
        # mae
        h_loss = K.mean(K.abs(y_true[1] - y_pred[1]))
        return rec_loss + (lambda_h * h_loss)
    
    def rec_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

    def h_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

    
    h_maxima_model = H_maxima_model(input_shape = input_shape, 
                                    train_dataloader = train_dataloader, 
                                    valid_dataloader = valid_dataloader, 
                                    test_dataloader = test_dataloader, 
                                    test_dataset = test_dataset, 
                                    EPOCHS=EPOCHS, 
                                    BATCH_SIZE = BATCH_SIZE, 
                                    loss = ["mean_squared_error","mean_absolute_error","mean_absolute_error","binary_crossentropy"], #{'hextrema': rec_loss, 'h_extrema_denoising_block2': h_loss}, #joint_loss, #["mean_squared_error","mean_squared_error"],  #"mean_squared_error", #"mean_absolute_error", #binary_crossentropy", #'mean_absolute_error',#'binary_crossentropy',  #'mean_absolute_error', #'binary_crossentropy', 
                                    loss_weights = [1.,0.,0.001,0.],
                                    #metrics = [rec_loss,h_loss],
                                    lr = LR,
                                    IMAGE_SAVE_PATH = IMAGE_SAVE_PATH,
                                    MODE = "only_hmaximalayer",
                                    resume = RESUME) #only_CNN pre_gaussian_blur post_gaussian_blur, post_processor , only_hmaximalayer, post_processor_aotoencoder
    #h_maxima_model.train()
    h_maxima_model.test()