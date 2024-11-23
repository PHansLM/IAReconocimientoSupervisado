# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import caffe 

# Ruta del dataset y archivo de etiquetas
dataset_path = "Data"
labels_file = "labels.csv"

# Lista de edades y géneros
lista_edades = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lista_generos = ['Masculino', 'Femenino']

# Cargar el archivo de etiquetas
data_labels = pd.read_csv(labels_file)

# Mapeo de edades a números
edad_map = {edad: idx for idx, edad in enumerate(lista_edades)}

# Mapeo de géneros a números
genero_map = {genero: idx for idx, genero in enumerate(lista_generos)}

# Procesar las imágenes y etiquetas
def cargar_datos():
    X = []  # Lista para imágenes procesadas
    y_edad = []  # Lista para etiquetas de edad
    y_genero = []  # Lista para etiquetas de género

    for idx, row in data_labels.iterrows():
        img_path = row['ruta_imagen']
        edad = row['edad']
        genero = row['genero']
        
        # Leer imagen
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Redimensionar a 227x227 (tamaño que espera el modelo)
        img = cv2.resize(img, (227, 227))
        # Normalizar (opcional, mejora resultados)
        img = img.astype("float32") / 255.0

        # Asignar la etiqueta de edad y género numérico
        edad_idx = edad_map[edad]  # Mapeo de edad
        genero_idx = genero_map[genero]  # Mapeo de género

        # Añadir la imagen y las etiquetas
        X.append(img)
        y_edad.append(edad_idx)
        y_genero.append(genero_idx)

    # Convertir listas a arrays de numpy
    X = np.array(X)
    y_edad = np.array(y_edad)
    y_genero = np.array(y_genero)

    # Convertir etiquetas a formato one-hot (para redes neuronales)
    y_edad = to_categorical(y_edad, num_classes=len(lista_edades))  # One-hot encoding para edades
    y_genero = to_categorical(y_genero, num_classes=len(lista_generos))  # One-hot encoding para géneros

    return X, y_edad, y_genero

# Cargar datos preprocesados
X, y_edad, y_genero = cargar_datos()

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_edad_train, y_edad_val, y_genero_train, y_genero_val = train_test_split(
    X, y_edad, y_genero, test_size=0.2, random_state=42)

# Cargar el modelo predefinido de Caffe
net_age = caffe.Net("programaDetector/age_deploy.prototxt", caffe.TRAIN)
net_age.copy_from("programaDetector/age_net.caffemodel")

net_gender = caffe.Net("programaDetector/gender_deploy.prototxt", caffe.TRAIN)
net_gender.copy_from("programaDetector/gender_net.caffemodel")

# Función de entrenamiento para un modelo
def entrenar_modelo(net, X_train, y_train, epochs=10, batch_size=32):
    for epoch in range(epochs):  # Número de épocas
        loss = 0
        for i in range(0, len(X_train), batch_size):  # Se usa batch_size para procesar en lotes
            # Preparar un lote de imágenes y etiquetas
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Redimensionar el lote para la red
            net.blobs['data'].data[...] = batch_X
            net.blobs['label'].data[...] = batch_y  # Aquí se usa la etiqueta correspondiente
            
            output = net.forward()
            loss += output['loss']  # Pérdida acumulada
            
            net.backward()
            net.update()

        print("Época {}, pérdida: {:.4f}".format(epoch + 1, loss / len(X_train)))

# Entrenar modelos de edad y género
print("Entrenando modelo de edad...")
entrenar_modelo(net_age, X_train, y_edad_train)

print("Entrenando modelo de género...")
entrenar_modelo(net_gender, X_train, y_genero_train)

# Guardar modelos entrenados
net_age.save("Modelos/age_modelo_entrenado.caffemodel")
net_gender.save("Modelos/gender_modelo_entrenado.caffemodel")
