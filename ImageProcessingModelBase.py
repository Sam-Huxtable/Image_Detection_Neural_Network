import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
#from sklearn.metrics import *
import itertools
import matplotlib.pyplot as plt


def DataGen():

	Train_Path = 'C:/Users/SHuxt/Downloads/POETdataset/POETdataset/Images/Train'
	Valid_Path = 'C:/Users/SHuxt/Downloads/POETdataset/POETdataset/Images/Valid'
	Test_Path = 'C:/Users/SHuxt/Downloads/POETdataset/POETdataset/Images/Test'
	
	TrainBatchs = ImageDataGenerator().flow_from_directory(Train_Path, target_size=(224, 224), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,)
	
	ValidBatchs = ImageDataGenerator().flow_from_directory(Train_Path, target_size=(224, 224), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,)
		
	TestBatchs = ImageDataGenerator().flow_from_directory(Train_Path, target_size=(224, 224), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,)
	
	return TrainBatchs, ValidBatchs, TestBatchs
	
def CNN(TrainBatchs, ValidBatchs, TestBatchs):
	
	#create model
	CNNmodel = Sequential()

	#add model layers
	CNNmodel.add(Conv2D(32, kernel_size=(3,3), activation= 'relu', input_shape=(224,224,3)))
	
	CNNmodel.add(Flatten())
	CNNmodel.add(Dense(2, activation= 'softmax'))
	
	CNNmodel.summary()
	CNNmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
	
	CNNmodel.fit_generator(TrainBatchs, steps_per_epoch = 10,
          validation_steps = 5,
          epochs = 5,
          verbose=2,
          validation_data = ValidBatchs)

def main():
	
	TrainBatchs, ValidBatchs, TestBatchs = DataGen()
	
	CNN(TrainBatchs, ValidBatchs, TestBatchs)



if __name__ == "__main__":
    main()