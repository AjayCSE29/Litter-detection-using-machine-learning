#This is a generic open-source code to train a CNN data model made by github.com/AjayCSE29
#This is the training file for the data model, this code can create a CNN data model from the given data set
#Make sure you have the latest version of the library Tensorflow to ensure no errors
#The latest version as of now (February 2026) supports GPU training by default.
#If you face any errors, try degrading or upgrading to my current tensorflow version, which is 2.20.0
#ADDITIONAL-NOTE: Increasing the BATCH_SIZE may increase the accuracy of the data-model but takes longer
import os
import tensorflow as tf #tensorflow version 2.20.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

DATASET_DIR = "xxxx" #your dataset directory
MODEL_PATH = "xxxx.h5" #your model path (or) path where your model will be stored

IMG_SIZE = (128, 128) #image size of the images used for training
BATCH_SIZE = 32  #BATCH_SIZE determines the quality (accuracy) of your data-model, the lessser the BATCH_SIZE, the better.
                  #But at the same time, if you're using a lower end system, increase the BATCH_SIZE so it'll not take a toll on your machine.
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#Declaring model and its specifications
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(train_data.class_indices), activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#Training of data
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

#Save model
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
