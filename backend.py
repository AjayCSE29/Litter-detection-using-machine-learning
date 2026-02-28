#This program is to use the trained CNN model 
#This is the main backend code which connects both frontend and the backend
#This project uses the flask library which serves as the connector between both ends
#The webpage will be boradcasted on a locally hosted ip: 127.0.0.1 with port 5000

from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

MODEL_PATH = "YOUR_MODEL.h5" #Path to where your trained CNN model is
IMG_PATH = "IMAGE" #The path of the image which will be verified
IMG_SIZE = (128, 128) #Size of the input image

model = tf.keras.models.load_model(MODEL_PATH) #Loading model

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory("dataset", target_size=IMG_SIZE, class_mode='categorical')
class_labels = list(generator.class_indices.keys())


@app.route("/")
def home():
    #The block which defines the main webpage
    return render_template("index.html") #Create a folder /template and save your webpage (HTML) as index.html

@app.route("/run-python")
def run_python():
    #Process of what happens afer you click the button to verify the object
    img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array) #Prediction of what the object might be
    predicted_index = np.argmax(predictions)

    predicted_label = class_labels[predicted_index] #Final prediction of what the object is

    return jsonify({
        #This block is to send the output to the frontend
        "output": f"{predicted_label}"
        #--output-- is the label which should be referenced in the main index.html file
    })

if __name__ == "__main__":
    #app.run(debug = False) if you're not running this program on Anaconda: Spyder
    app.run(debug=True, use_reloader=False)
