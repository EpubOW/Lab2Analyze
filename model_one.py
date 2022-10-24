#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('D:\GitHub\Lab2Analyze\Lab2Analyze\Dataset_Food'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[23]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import tensorflow as tf
import json
import os
from keras.models import load_model


# In[24]:


im_height = 224
im_width = 224
batch_size = 128
epochs = 1


# In[26]:


image_path = "D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/"  # flower data set path
train_dir = image_path + "training"
validation_dir = image_path + "validation"
test_dir = image_path + "evaluation"
# data generator with data augmentation
train_image_generator = ImageDataGenerator( rescale=1./255, 
                                            rotation_range=40, 
                                            width_shift_range=0.2,
                                            height_shift_range=0.2, 
                                            zoom_range=0.2,
                                            horizontal_flip=True, 
                                            fill_mode='nearest')


validation_image_generator = ImageDataGenerator(rescale=1./255)


test_image_generator = ImageDataGenerator(rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
total_val = val_data_gen.n

test_data_gen = test_image_generator.flow_from_directory( directory=test_dir,
                                                          target_size=(im_height, im_width))

total_test = test_data_gen.n


# In[27]:


covn_base = tf.keras.applications.xception.Xception(weights='imagenet',include_top=False)
covn_base.trainable = True

for layers in covn_base.layers[:-32]:
    layers.trainable = False

model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(2))


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])


# In[28]:


Early_sp = EarlyStopping(monitor = 'val_accuracy', patience = 5,restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=[Early_sp,reduce_lr])

model.save('model.h5')