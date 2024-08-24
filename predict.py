import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

model = keras.models.load_model('mobilenet.keras')

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image


def predict_image(image_path, class_names):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0]
    return class_names[np.argmax(prediction)]

class_names = sorted(os.listdir('FastFoodClassificationV2/Train'))
image_path = '/home/paulo/avanti-bootcamp-ML1/avanti-bootcamp-ML/FastFoodClassificationV2/Test/Taco/Taco-Test (14).jpeg'

predicted_class = predict_image(image_path, class_names)
print(f'predict: {predicted_class}')
