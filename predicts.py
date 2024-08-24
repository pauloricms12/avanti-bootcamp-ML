import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model('mobilenet.keras')

test_dir = 'FastFoodClassificationV2/Test'

class_names = sorted(os.listdir(test_dir))  

y_true = []
y_pred = []

for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) 
        image = image.astype('float32') / 255.0  
        image = np.expand_dims(image, axis=0)  
        
        pred_probs = model.predict(image)
        pred_class = np.argmax(pred_probs, axis=1)[0]
        
        y_true.append(class_names.index(class_name))
        y_pred.append(pred_class)

conf_matrix = confusion_matrix(y_true, y_pred)

conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

total_sum = np.sum(conf_df)

print(f"A soma de todos os valores na matriz de confusão é: {total_sum}")

conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_df_percent = pd.DataFrame(conf_matrix_percent, index=class_names, columns=class_names)

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão (Valores Absolutos)')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

plt.subplot(1, 2, 2)
sns.heatmap(conf_df_percent, annot=True, fmt='.2f', cmap='Blues')
plt.title('Matriz de Confusão (Percentual)')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

plt.show()

