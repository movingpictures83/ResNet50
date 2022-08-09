import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#import cv2

import os
################################################################################################################################
# FOR PARALLELISM
from tensorflow.keras import backend as K
import tensorflow as tf
NUM_PARALLEL_EXEC_UNITS = 20
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)

######################################################################################
# ALL OF THESE WILL BE PARAMETERS
#os.environ["OMP_NUM_THREADS"] = "10"
#os.environ["KMP_BLOCKTIME"] = "30"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# Fixed for our Cats & Dogs classes
#NUM_CLASSES = 2

# Fixed for Cats & Dogs color images
#CHANNELS = 3

#IMAGE_RESIZE = 224
#RESNET50_POOLING_AVERAGE = 'avg'
#DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
#LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
#NUM_EPOCHS = 10
#EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
#STEPS_PER_EPOCH_TRAINING = 10
#STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
#BATCH_SIZE_TRAINING = 100
#BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
#BATCH_SIZE_TESTING = 1
################################################################################################


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense



class ResNet50Plugin:
    def input(self, inputfile):
       ######################################################################
       # READ FILE OF TAB-DELIMITED KEYWORD-VALUE PAIRS
       self.parameters = dict()
       infile = open(inputfile, 'r')
       for line in infile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]
       ######################################################################

       os.environ["OMP_NUM_THREADS"] = self.parameters["OMP_NUM_THREADS"]
       os.environ["KMP_BLOCKTIME"] = self.parameters["KMP_BLOCKTIME"]
       os.environ["KMP_SETTINGS"] = self.parameters["KMP_SETTINGS"]
       os.environ["KMP_AFFINITY"] = self.parameters["KMP_AFFINITY"]

       # This file, also, is a parameter now
       # Will call it inputfile
       #resnet_weights_path = 'input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
       self.resnet_weights_path = self.parameters["inputfile"]
       # Note I now made this the input file
       # Because I need it in other functions, I preface with "self"

    def run(self):
       print(self.resnet_weights_path)
       self.model = Sequential()
       self.model.add(ResNet50(include_top = False, pooling = self.parameters["RESNET50_POOLING_AVERAGE"], weights = self.resnet_weights_path))

       self.model.add(Dense(int(self.parameters["NUM_CLASSES"]), activation = self.parameters["DENSE_LAYER_ACTIVATION"]))
       self.model.layers[0].trainable = False
       self.model.summary()
       from tensorflow.python.keras import optimizers
       #sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
       sgd = optimizers.SGD(lr = float(self.parameters["lr"]), decay = float(self.parameters["decay"]), momentum = float(self.parameters["momentum"]), nesterov = True)
       self.model.compile(optimizer = sgd, loss = self.parameters["OBJECTIVE_FUNCTION"], metrics = [self.parameters["LOSS_METRICS"]])
 
       from tensorflow.keras.applications.resnet50 import preprocess_input
       from tensorflow.keras.preprocessing.image import ImageDataGenerator
       image_size = int(self.parameters["IMAGE_RESIZE"])

       data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

       # Need in output(), preface with 'self'
       self.train_generator = data_generator.flow_from_directory(
       #'input/trainvalidfull4keras/train',
       self.parameters["TRAIN_FOLDER"],
       target_size=(image_size, image_size),
       batch_size=int(self.parameters["BATCH_SIZE_TRAINING"]),
       class_mode='categorical')

       self.validation_generator = data_generator.flow_from_directory(
       #'input/trainvalidfull4keras/valid',
       self.parameters["VALIDATE_FOLDER"],
       target_size=(image_size, image_size),
       batch_size=int(self.parameters["BATCH_SIZE_VALIDATION"]),
       class_mode='categorical') 


    def output(self, filename):
        from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
        cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = int(self.parameters["EARLY_STOP_PATIENCE"]))
        #cb_checkpointer = ModelCheckpoint(filepath = 'working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
        cb_checkpointer = ModelCheckpoint(filepath = self.parameters["BEST"], monitor = 'val_loss', save_best_only = True, mode = 'auto')

        fit_history = self.model.fit_generator(
        self.train_generator,
        steps_per_epoch=int(self.parameters["STEPS_PER_EPOCH_TRAINING"]),
        epochs = int(self.parameters["NUM_EPOCHS"]),
        validation_data=self.validation_generator,
        validation_steps=int(self.parameters["STEPS_PER_EPOCH_VALIDATION"]),
        callbacks=[cb_checkpointer, cb_early_stopper]
        )
        plt.style.use("ggplot")
        plt.figure()
        print(fit_history.history)
        plt.plot(np.arange(0, len(fit_history.history["loss"])), fit_history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, len(fit_history.history["val_loss"])), fit_history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, len(fit_history.history["acc"])), fit_history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, len(fit_history.history["val_acc"])), fit_history.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(filename)

