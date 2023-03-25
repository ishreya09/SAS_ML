import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

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


def model_prediction(image):
    
    SAR_CNN = tf.keras.models.load_model('CNN_models/SAR_CNN.h5',
                                     custom_objects={'f1_score':f1_score,
                                                     'recall_m':recall_m,
                                                     'precision_m':precision_m
                                                     }
                                    )

    pred = np.argmax(SAR_CNN.predict(image[tf.newaxis,:,:,:]))
    prd = int(pred.ravel())
    return pred

image_paths=  ["sen12flood\sen12floods_s1_source\sen12floods_s1_source\sen12floods_s1_source_0_2019_02_18" ]
img=tf.data.Dataset.from_tensor_slices(image_paths)
model_prediction(img)
