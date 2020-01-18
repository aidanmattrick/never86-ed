#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.applications import ResNet50, Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import multi_gpu_model
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session  


# In[2]:


run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


# In[3]:


"""
Setup multi GPU usage

Example usage:
model = Sequential()
...
multi_model = multi_gpu_model(model, gpus=num_gpu)
multi_model.fit()

About memory usage:
https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
"""

# IMPORTANT: Tells tf to not occupy a specific amount of memory
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
sess = tf.Session(config=config)  

set_session(sess)  # set this TensorFlow session as the default session for Keras.


local_device_protos = device_lib.list_local_devices()
gpus = [x.name for x in local_device_protos if x.device_type    == 'GPU']
print("GPUs:" + str(gpus))
num_gpu = len(gpus)
print('Amount of GPUs available: %s' % num_gpu)


# In[4]:


IMAGE_COUNT=3699
EPOCHS=10
BATCH_SIZE=16
STEPS_PER_EPOCH=IMAGE_COUNT/BATCH_SIZE
TARGET_SIZE=(200, 200)
COLOR_MODE='rgb'
#STEPS_PER_EPOCH=25000

BASE_PATH=Path('/efs/notebooks/volume')
TRAIN_PATH=Path(BASE_PATH, 'train-cropped-bw')

DATETIME_STR = datetime.now().strftime('%Y%m%d-%H%M')
MODEL_SAVE_PATH=Path(BASE_PATH, 'model-' + DATETIME_STR + '.h5')


# In[5]:


MODEL_SAVE_PATH


# In[6]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.20
)


# In[7]:


train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode=COLOR_MODE
)


# In[8]:


CATEGORY_COUNT=len(train_generator.class_indices.keys())


# In[9]:


#list(train_generator.class_indices.keys())
' '.join(list(train_generator.class_indices.keys()))


# In[10]:


test_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        #color_mode=COLOR_MODE
)


# In[11]:


base_model = ResNet50(include_top=False, input_shape=(200, 200, 3), weights= "imagenet")
#base_model = Xception(include_top=False, input_shape=(299, 299, 3))


# In[12]:


#Tyson recommends put another 200 node dense layer, use ReLu, 
#it will improve accuracy. 
#For curiousity sake add another 200 node dense layer. 
#Reduce epochs to 10
#Don't play with batch size
#See where test value is at first... He recommends go straight to test... 
#Overfitting is big drop in accuracy... i.e. anything more than 15 basis points
#Buttt late 60 to 80 is good
#Tinker with learning rate if you are doing extra layers... 
#Make sure to take out for loop that is freezing previous layers
#Play with hyperparameters
#Stay with Adam


# In[13]:


output = Flatten()(base_model.input)


# In[14]:


output = Dense(2000, activation='relu')(output)
output = Dropout(rate = 0.25)(output)
output = Dense(2000, activation='relu')(output)
#output = Dense(1000, activation='relu')(output)


# In[15]:


output = Dense(CATEGORY_COUNT, activation='softmax')(output)


# In[16]:


for layer in base_model.layers:
    layer.trainable = False
model = Model(inputs=base_model.input, outputs=[output])
#parallel_model = model 
parallel_model = multi_gpu_model(model, gpus=8, cpu_merge=False)


# In[17]:


opt = Adam(lr=0.00001)
import functools
top2_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)
top2_acc.__name__ = 'top2_acc'
parallel_model.compile(loss='categorical_crossentropy', optimizer = opt , metrics=['accuracy', top2_acc])


# In[18]:


parallel_model.fit_generator(
  train_generator,
  steps_per_epoch=STEPS_PER_EPOCH,
  epochs=EPOCHS,
  workers=64,
  verbose=1
)
#        validation_data=validation_generator,
#        validation_steps=800)


# In[19]:


parallel_model.evaluate_generator(
  test_generator,
  workers=64,
  verbose=1,
  steps=STEPS_PER_EPOCH
)


# In[20]:


model.save(str(MODEL_SAVE_PATH))


# In[ ]:




