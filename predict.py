import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
import os
import rasterio as rio
import cv2

def load_raster(filepath): # filepath of the raster file to be loaded
    '''load a single band raster'''
    with rio.open(filepath) as file: 
        # the squeeze method is called on the resulting array to remove any singleton dimensions 
        # (i.e., dimensions with size 1). This is done using the axis=0 argument, 
        # which tells squeeze to remove any singleton dimensions along the first axis.  
        raster = file.read().squeeze(axis=0)

        
    #we aregetting back the 2D image from singleton(1D)
    return raster

def load_s1_tiffs(folder,
                  scaling_values=[50.,100.]):
    images = []
    i = 0
    for im in sorted(os.listdir(folder)):
         
        if im.rsplit('.',maxsplit=1)[1] == 'tif':
            
            path = folder + '/' + im
            band = load_raster(path)
            band = band / scaling_values[i]
            
            band = cv2.resize(band,
                              CFG.img_size)
            
            images.append(band)
            i+=1 
                    
    return np.dstack(images)


def load_s2_tiffs(folder,
                  scaling_value=10000.):
    images = []
    for im in sorted(os.listdir(folder)):
        if im.rsplit('.',maxsplit=1)[1] == 'tif':    
            path = folder + '/' + im
            band = load_raster(path)
            band = band/ scaling_value
            
            band = cv2.resize(band,CFG.img_size)
            images.append(band)   

    return np.dstack(images)
                    
    
def tf_load_s1(path):    
    path = path.numpy().decode('utf-8')
    return load_s1_tiffs(path)
    
    

def tf_load_s2(path):    
    path = path.numpy().decode('utf-8')
    return load_s2_tiffs(path)

    
def process_image_s1(filename):
    '''function for preprocessing in tensorflow data'''
    
    return tf.py_function(tf_load_s1, 
                          [filename], 
                          tf.float32)



def process_image_s2(filename):
    '''function for preprocessing in tensorflow data'''
    
    return tf.py_function(tf_load_s2, 
                          [filename], 
                          tf.float32)



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class CFG:
    """
    This class provides a set of parameters and constants that may be used throughout a machine learning 
    pipeline for image classification, specifically in the context of identifying flooded areas.
    """
    seed = 3 # random initialization of weights in a machine learning model
    img_size = (256,256) # representing the dimensions of an image, specifically 256 x 256 pixels.
    BATCH_SIZE = 3 #  representing the number of samples that will be fed to a machine learning model during training.
    Autotune = tf.data.AUTOTUNE # a constant value from the tf.data.AUTOTUNE module that enables dynamic 
    # allocation of computational resources to improve performance.
    validation_size = 0.2 # a float value of 0.2 representing the fraction of the training dataset to be used for validation during training.
    class_dict= {0:'No Flooding', 
                 1: 'Flooding'}
    
    test_run = False # in training mode

def get_tf_dataset(image_paths,
                   labels=None, # put none for test data set
                   image_processing_fn=None,
                   augment_fn = None
                  ):
    
    
    '''returns a tf dataset object
    Inputs: 
    image_paths : paths to images
    labels: labels of each image
    image_processing_fn:  function to load and preprocess images 
    augment_fn : function to augment images '''
    
    #seperate datasets
    if labels is not None:
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    
    
    
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    #load images 
    image_dataset = image_dataset.map(image_processing_fn,
                                      num_parallel_calls=tf.data.AUTOTUNE)
     
    if augment_fn is not None:
        
        image_dataset = image_dataset.map(augment_fn,
                                          num_parallel_calls=tf.data.AUTOTUNE)
     
    
    if labels is not None:
        return tf.data.Dataset.zip((image_dataset,labels_dataset))
    
    
    return image_dataset



def optimize_pipeline(tf_dataset,
                      batch_size = CFG.BATCH_SIZE,
                      Autotune_fn = CFG.Autotune,
                      cache= False,
                      batch = True):
    
    
    
    # prefetch(load the data with cpu,while gpu is training) the data in memory 
    tf_dataset = tf_dataset.prefetch(buffer_size=Autotune_fn)  
    if cache:
        tf_dataset = tf_dataset.cache()                        # store data in RAM  
        
    tf_dataset =  tf_dataset.shuffle(buffer_size=50)         # shuffle 
    
    if batch:
        tf_dataset = tf_dataset.batch(batch_size)              #split the data in batches  
    
    return tf_dataset

# Sentinel 1 dataset (not using augmentation here)


SAR_CNN = tf.keras.models.load_model('CNN_models/SAR_CNN.h5',
                                     custom_objects={'f1_score': f1_score,
                                                     'precision_m': precision_m,
                                                     'recall_m': recall_m})

def model_prediction(image):
    

    pred = np.argmax(SAR_CNN.predict(image[tf.newaxis,:,:,:]))
    # prd = int(pred.ravel())
    print(pred)
    return pred

def set_data(s_data):
    S1_dataset = optimize_pipeline(tf_dataset=get_tf_dataset(image_paths = s_data.image_dir.values,
                                                labels = s_data.label,
                                                image_processing_fn = process_image_s1),
                                    
                                    batch_size = 3 * CFG.BATCH_SIZE)
    return S1_dataset

data= set_data(pd.read_csv('TimewiseCSV\\2018-12-16_s1.csv'))
for images,labels in data:
    # print(images.shape)
    # print(labels.shape)
    # print(labels)
    # print("IMG",images)
    for k in range(len(images)):
        # print("img",images[k])
        try:
            print(model_prediction(images[k]))
        except:
            pass
