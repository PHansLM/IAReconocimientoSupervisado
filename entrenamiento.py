import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog

def obtener_nombre_archivo():
    root = tk.Tk()
    root.withdraw()
    nombre_archivo = simpledialog.askstring("Guardar Modelo", "Ingresa el nombre del archivo:")
    root.destroy()
    return nombre_archivo

direccionDatos = "D:/progra/IA/proyectoFinal/Data"
listaPersonas = os.listdir(direccionDatos)
print('Lista de personas: ' + str(listaPersonas))

etiquetas = []
datosCaras = []
etiqueta = 0
tamano_imagen = (150, 150)

for nameDir in listaPersonas:
    direccionPersona = direccionDatos + '/' + nameDir
    print(f'Escaneando imágenes en {direccionPersona}')

    for fileName in os.listdir(direccionPersona):
        print('Procesando rostro', nameDir + '/' + fileName)
        image = cv2.imread(direccionPersona + '/' + fileName, 0)
        if image is None:
            print(f"Advertencia: No se pudo cargar la imagen {fileName}. Saltando esta imagen.")
            continue
        image = cv2.resize(image, tamano_imagen)
        
        etiquetas.append(etiqueta)
        datosCaras.append(image)

    etiqueta += 1

detectorRostros = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando el modelo LBPH...")

detectorRostros.train(datosCaras, np.array(etiquetas))

nombreArchivo = obtener_nombre_archivo()
if nombreArchivo:
    rutaModelo = f'D:/progra/IA/proyectoFinal/Modelos/{nombreArchivo}.xml'
    detectorRostros.write(rutaModelo)
    print(f"Modelo almacenado como {rutaModelo}")
else:
    print("No se proporcionó un nombre de archivo.")
