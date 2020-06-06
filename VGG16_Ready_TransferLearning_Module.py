#!/usr/bin/env python
# coding: utf-8
#importing necessary basic libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.image import imread


#arrange your data in the desired drive drive->dataset->test->class~1->data,,..class~n->
#arrange your data in the desired drive drive->dataset->train->class~1->data,,..class~n->


#reading files for getting the shape of data in training

data_test_dir="E://dataset//test"
data_train_dir="E://dataset//train"

#dimensions for dataset

for i in os.listdir(data_train_dir):
    for j in os.listdir(data_train_dir+"//"+i):
        dim1.append((imread("E://dataset2//train//"+i+"//"+j)).shape[0])
        dim2.append((imread("E://dataset2//train//"+i+"//"+j)).shape[1])
        

print(np.mean(dim1))
print(np.mean(dim2))

#importing vgg16 architecture

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


#initialising model VGG16.
#Setting include_top=False basically this helps us in getting away the original 1000 class output available.
#weights here refer to pretrained ones
#it accepts an input_shape=[224,224,3]
#you can change the input_shape but the model discourages it  because the trained weights are actually better for 224,224,3

vgg = VGG16(input_shape=[224,224] + [3], weights='imagenet', include_top=False)

#stacking layers on our model for our own customized output
#the dense layer has 4 neurons feel free to change according to the no of classes


y=Flatten()(vgg.output)
y=Dense(4,activation="softmax")(y)


#setting the layers to be non-trainable 
#the actual model weights took 3 weeks on nvidia titan gpu


for layers in vgg.layers:
    layers.trainable=False

model=Model(inputs=vgg.input, output=y)

#visualize the model

print(model.summary())

#you can adjust shear,zoom,color_mode,etc in ImageDataGenerator




from keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(rescale=1/255,fill_mode="nearest")
test_gen=ImageDataGenerator(rescale=1/255,fill_mode="nearest")
test_set = train_gen.flow_from_directory('E://dataset2//test',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
training_set = train_gen.flow_from_directory('E://dataset2//train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


#compiling

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)





from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
mc=ModelCheckpoint(" weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_accuracy',mode="max",verbose=1, save_best_only=True)
es=EarlyStopping(patience=3,monitor='val_loss')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,callbacks=[es,mc]
)

from tensorflow.keras.models import load_model
model.save('mausam.h5')

plt.plot(model.history.history)




