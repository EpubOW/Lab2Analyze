import numpy as np # linear algebra

# In[2]:


import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model = load_model('model.h5')

test_datagen = ImageDataGenerator(
                    rescale = 1./255)

test_generator = test_datagen.flow_from_directory(directory='D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/evaluation',
                                                   target_size=(128,128),
                                                   classes=['food','non_food'],
                                                   class_mode='binary')

with open("Dataset/result_first.txt", "w") as file:
    file.write(repr(model.evaluate_generator(test_generator,steps=len(test_generator))))