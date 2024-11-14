import cv2
import os
import tkinter as tk
from tkinter import filedialog

def seleccionar_archivo_modelo():
    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo del modelo",
        filetypes=[("Archivos XML", "*.xml")]
    )
    root.destroy()
    return archivo

def seleccionar_archivo_video():
    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo de video",
        filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    root.destroy()
    return archivo

direccionDatos = "D:/progra/IA/proyectoFinal/Data"
direccionImagenes = os.listdir(direccionDatos)
print('Lista de personas: ' + str(direccionImagenes))

# Cargar el modelo de reconocimiento facial LBPH
detectorRostros = cv2.face.LBPHFaceRecognizer_create()
archivo_modelo = seleccionar_archivo_modelo()
if archivo_modelo:
    detectorRostros.read(archivo_modelo)
    print(f"Modelo cargado desde: {archivo_modelo}")
else:
    print("No se seleccionó ningún archivo de modelo.")

# Seleccionar y abrir el archivo de video
archivo_video = seleccionar_archivo_video()
if archivo_video:
    captura = cv2.VideoCapture(archivo_video)
    print(f"Video cargado desde: {archivo_video}")
else:
    print("No se seleccionó ningún archivo de video.")

# Cargar el clasificador de rostros
clasificadorRostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = captura.read()
    if not ret:
        break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAuxiliar = gris.copy()

    # Detectar rostros en el cuadro actual
    caras = clasificadorRostros.detectMultiScale(gris, 1.3, 5)
    for (x, y, w, h) in caras:
        rostro = frameAuxiliar[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (100, 100), interpolation=cv2.INTER_CUBIC)
        resultado = detectorRostros.predict(rostro)
        cv2.putText(frame, '{}'.format(resultado), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if resultado[1] < 50:  # Umbral ajustado para LBPH
            cv2.putText(frame, '{}'.format(direccionImagenes[resultado[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:  # Presiona Esc para salir
        break

captura.release()
cv2.destroyAllWindows()
