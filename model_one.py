#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra

# In[2]:


import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# In[7]:


train_datagen = ImageDataGenerator(
                    rescale = 1./255)

train_generator = train_datagen.flow_from_directory(directory='D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training',
                                                   target_size=(128,128),
                                                   classes=['food','non_food'],
                                                   class_mode='binary')


# In[8]:


valid_datagen = ImageDataGenerator(
                    rescale = 1./255)

valid_generator = valid_datagen.flow_from_directory(directory='D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/validation',
                                                   target_size=(128,128),
                                                   classes=['food','non_food'],
                                                   class_mode='binary')


# In[9]:

# print(type(test_datagen))


# In[10]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,kernel_initializer='he_normal',kernel_size=(3,3),input_shape=(128,128,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128,kernel_initializer='he_normal',kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(256,kernel_initializer='he_normal',kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))


# # In[11]:


early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)


# In[12]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


Y = model.fit_generator(train_generator, epochs=1,validation_data=valid_generator)


model.save('model.h5')




# # In[15]:



