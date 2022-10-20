#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob as gb
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from matplotlib.image import imread
import os
import glob
import matplotlib.image as mpimg
from pathlib import Path
import seaborn as sns
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks  import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix


# In[8]:


from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
import tensorflow as tf
import random
from  sklearn.utils import shuffle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.layers import Input, Add, Dense,GlobalAvgPool2D
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.models import Model
from keras.layers import Input, Add, Dense, Concatenate, AvgPool2D, Dropout,BatchNormalization,  GlobalAveragePooling2D


# In[9]:


# plt.figure(figsize=(10,10))
# food_images="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training/food/"
# for i in range(12):
#     file=random.choice(os.listdir(food_images))
#     food_image_path=os.path.join(food_images,file)
#     img=mpimg.imread(food_image_path)
#     ax=plt.subplot(3,4,i+1)
#     plt.imshow(img)
# plt.show()


# In[10]:


# plt.figure(figsize=(10,10))
# non_food_images="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training/non_food/"
# for i in range(12):
#     file=random.choice(os.listdir(non_food_images))
#     non_food_image_path=os.path.join(non_food_images,file)
#     img=mpimg.imread(non_food_image_path)
#     ax=plt.subplot(3,4,i+1)
#     plt.imshow(img)
# plt.show()


# In[11]:


image_path=Path("D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training/")
img_path="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training/"
image_size=(224,224)
train_path="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/training/"
valid_path="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/validation/"
test_path="D:/GitHub/Lab2Analyze/Lab2Analyze/Dataset_Food/evaluation/"


# In[12]:


class_names = os.listdir(image_path)
print(class_names)
print("Number of classes : {}".format(len(class_names)))


# In[13]:


numberof_images={}
for class_name in class_names:
    numberof_images[class_name]=len(os.listdir(img_path+"/"+class_name))
images_each_class=pd.DataFrame(numberof_images.values(),index=numberof_images.keys(),columns=["Number of images"])
images_each_class


# In[14]:


batch_size=30
traindata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,
                                                                    shear_range=0.2, horizontal_flip=True,validation_split=0.2,fill_mode='nearest')

validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[15]:


train_data_generator=traindata_generator.flow_from_directory(train_path,batch_size=batch_size,class_mode="categorical",
                                                           target_size=(224,224),color_mode="rgb",shuffle=True )
valid_data_generator=validdata_generator.flow_from_directory(valid_path,batch_size=batch_size,class_mode="categorical",
                                                           target_size=(224,224),color_mode="rgb",shuffle=True )
test_data_generator=testdata_generator.flow_from_directory(test_path,batch_size=batch_size,class_mode="categorical",
                                                           target_size=(224,224),color_mode="rgb",shuffle=False )


# In[16]:


class_dict = train_data_generator.class_indices
class_list = list(class_dict.keys())
class_list


# In[17]:


train_number=train_data_generator.samples
valid_number=valid_data_generator.samples


# In[18]:


dense121_model= tf.keras.applications.densenet.DenseNet121(weights='imagenet',include_top=False, input_shape=(224,224, 3))
x= dense121_model.output
x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x) 
x= Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x) 

x= Dropout(0.5)(x)
prediction= Dense(2, activation = 'softmax')(x)
model= Model(inputs= dense121_model.input, outputs= prediction)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[19]:


tensor_board=TensorBoard(log_dir="logs")
check_point=ModelCheckpoint("denseNet121.h5",monitor="val_accuracy",mode="auto",verbose=1,save_best_only=True)


# In[20]:


reduce_lr=ReduceLROnPlateau(monitor="val_accuracy",factor=0.3,patience=50,min_delta=0.001,mode="auto",verbose=1)


# In[26]:


history= model.fit(train_data_generator, 
                   steps_per_epoch=train_number//batch_size, 
                   validation_data= valid_data_generator, 
                   validation_steps= valid_number//batch_size,
                   shuffle=True, 
                   
                   epochs =1, 
                   batch_size = 30,callbacks=[tensor_board,check_point,reduce_lr])


# In[27]:


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[30]:

with open("Dataset/result_second.txt", "w") as file:
    file.write(repr(model.evaluate_generator(test_data_generator,steps=len(test_data_generator))))


