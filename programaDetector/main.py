import cv2  # Importar la librería OpenCV para trabajar con visión por computadora

# Archivos de configuración y modelos para la detección de rostros
prototipo_rostro = "D:/progra/IA/proyectoFinal/programaDetector/opencv_face_detector.pbtxt"
modelo_rostro = "D:/progra/IA/proyectoFinal/programaDetector/opencv_face_detector_uint8.pb"

# Archivos de configuración y modelos para la estimación de edad
prototipo_edad = "D:/progra/IA/proyectoFinal/programaDetector/age_deploy.prototxt"
modelo_edad = "D:/progra/IA/proyectoFinal/programaDetector/age_net.caffemodel"

# Archivos de configuración y modelos para la estimación de género
prototipo_genero = "D:/progra/IA/proyectoFinal/programaDetector/gender_deploy.prototxt"
modelo_genero = "D:/progra/IA/proyectoFinal/programaDetector/gender_net.caffemodel"

# Función para detectar rostros y dibujar rectángulos alrededor de ellos
def dibujar_rectangulos(red_rostros, cuadro):
    # Obtener dimensiones del cuadro (ancho y alto)
    ancho_cuadro = cuadro.shape[1]
    alto_cuadro = cuadro.shape[0]
    
    # Preprocesar el cuadro para la red neuronal
    blob = cv2.dnn.blobFromImage(cuadro, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    red_rostros.setInput(blob)
    
    # Lista para almacenar las coordenadas de los rectángulos detectados
    rectangulos = []
    detecciones = red_rostros.forward()
    
    # Iterar sobre las detecciones
    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]  # Obtener la confianza de la detección
        if confianza > 0.7:  # Umbral de confianza
            # Calcular las coordenadas del rectángulo
            x1 = int(detecciones[0, 0, i, 3] * ancho_cuadro)
            y1 = int(detecciones[0, 0, i, 4] * alto_cuadro)
            x2 = int(detecciones[0, 0, i, 5] * ancho_cuadro)
            y2 = int(detecciones[0, 0, i, 6] * alto_cuadro)
            
            # Agregar el rectángulo a la lista y dibujarlo en el cuadro
            rectangulos.append([x1, y1, x2, y2])
            cv2.rectangle(cuadro, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return cuadro, rectangulos

# Cargar las redes neuronales para detección de rostros, edad y género
red_rostros = cv2.dnn.readNet(modelo_rostro, prototipo_rostro)
red_edad = cv2.dnn.readNet(modelo_edad, prototipo_edad)
red_genero = cv2.dnn.readNet(modelo_genero, prototipo_genero)

# Iniciar la captura de video desde la cámara
video = cv2.VideoCapture(0)

# Listas de etiquetas para edades y géneros
lista_edades = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lista_generos = ['Masculino', 'Femenino']

# Valores de normalización para el modelo
valores_modelo = (78.4263377603, 87.7689143744, 114.895847746)

# Bucle principal para procesar el video en tiempo real
while True:
    ret, cuadro = video.read()  # Leer un cuadro de la cámara
    cuadro, rectangulos = dibujar_rectangulos(red_rostros, cuadro)  # Detectar rostros
    
    # Procesar cada rostro detectado
    for rectangulo in rectangulos:
        rostro = cuadro[rectangulo[1]:rectangulo[3], rectangulo[0]:rectangulo[2]]  # Recortar el rostro
        blob = cv2.dnn.blobFromImage(cuadro, 1.0, (227, 227), valores_modelo, swapRB=False)
        
        # Predicción de género
        red_genero.setInput(blob)
        prediccion_genero = red_genero.forward()
        genero = lista_generos[prediccion_genero[0].argmax()]
        
        # Predicción de edad
        red_edad.setInput(blob)
        prediccion_edad = red_edad.forward()
        edad = lista_edades[prediccion_edad[0].argmax()]
        
        # Etiqueta para mostrar en el cuadro
        etiqueta = "{},{}".format(genero, edad)
        cv2.putText(cuadro, etiqueta, (rectangulo[0], rectangulo[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2)
    
    # Mostrar el cuadro con las detecciones
    cv2.imshow("Edad - Genero", cuadro)
    tecla = cv2.waitKey(1)
    if tecla == ord('q'):  # Salir si se presiona la tecla 'q'
        break

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
