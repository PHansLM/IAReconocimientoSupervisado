import tkinter as tk
import cv2
import os
import imutils
from tkinter import filedialog

root = tk.Tk()
root.geometry("400x200")
root.title("Entrada de Nombre")

label = tk.Label(root, text="Ingresa tu nombre:", font=("Poppins", 12))
label.pack(pady=10)

entradaNombre = tk.Entry(root, font=("Poppins", 14), width=30)
entradaNombre.pack(pady=10)

def obtener_nombre():
    global nombrePersona
    nombre = entradaNombre.get()
    if nombre:
        nombrePersona = nombre
    else:
        nombrePersona = ""
    root.quit()

boton = tk.Button(root, text="Aceptar", command=obtener_nombre, font=("Poppins", 12))
boton.pack(pady=10)

root.mainloop()

direccionDatos = "D:/progra/IA/proyectoFinal/Data"
direccionPersona = direccionDatos + "/" + nombrePersona
print(direccionPersona)
if not os.path.exists(direccionPersona):
    print('Se creo la carpeta: ' + direccionPersona)
    os.makedirs(direccionPersona)

videoArchivo = filedialog.askopenfilename(title="Selecciona un archivo de video", filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov")])

if videoArchivo:
    videoCaptura = cv2.VideoCapture(videoArchivo)
else:
    print("No se seleccionó un archivo.")

clasificarRostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
contador = 0

while True:
    ret, frame = videoCaptura.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAuxiliar = frame.copy()

    caras = clasificarRostros.detectMultiScale(gris,1.3,5)
    for(x,y,w,h) in caras:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = frameAuxiliar[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(direccionPersona + '/rostro_{}.jpg'.format(contador),rostro)
        contador = contador + 1
    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27 or contador >= 300:
        break
videoCaptura.release()
cv2.destroyAllWindows()