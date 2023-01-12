#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 13 - El ordenador saber leer
#
# Módulos necesarios:
#   TENSORFLOW 1.13.1
#   KERAS 2.2.4
#   OPENCV 3.4.5.20
#   PYTTSX3 2.7.1
#   SCIKIT-LEARN 0.21.1
#   NUMPY 1.16.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


import cv2
import numpy as np

#Module de paroles
import pyttsx3 as pyttsx

#Module Keras permettant l'utilisation de notre réseau de neurones
from keras.models import load_model

#Module de gestion des processus
import threading

class Etapa3:

    def __init__(self):
        #De manera predeterminada se activa la lectura de letra en voz alta
        self.lectureActivee = True

        #Tiempo de espera en segundos entre cada lectura de letra en voz alta
        self.duraciónDesactivacionLecturaDeLetra = 5

    #función de reactivación de la lectura de letra en voz alta
    def activacionLectura(self):
        print('Activación de la lectura de letras')
        global lecturaActivada
        lecturaActivada=True

    def dimensiones(self):
        #dimensiones de la zona de escritura
        self.zonaEscrituraLargoMin = 540
        self.zonaEscrituraLargoMax = 590
        self.zonaEscrituraAnchoMin = 300
        self.zonaEscrituraAnchoMax = 340

    def inicioVoz(self):
        #Inicialización de la voz
        print('Initialización de la voz')
        self.engine = pyttsx.init()

    def eleccionVoz(self):
        #Elección de la voz en español
        self.voice = self.engine.getProperty('voices')[0]
        self.engine.setProperty('voice', self.voice.id)

    def pruebaVoz(self):
        #prueba de la voz
        self.engine.say('Modo lectura de letras activado')
        self.engine.runAndWait()

    def aprendizaje(self):
        #Inicialización del modelo de aprendizaje
        print('Inicialización del modelo de aprendizaje')

        #Carga del modelo entrenado
        self.cnn_model = load_model('modelo/modelo_caso_practicoV2.h5')
        self.kernel = np.ones((5, 5), np.uint8)

        #Tabla de letras con su número
        self.letras = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
                    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
                    21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '-'}



        #De manera predeterminada se elige que se detecte la letra Z
        self.prediccion = 26

        #De manera predeterminada no se hace ninguna predicción.
        self.letraPredicha = False

    def inicioWebcam(self):
        #Inicialización de la webcam
        print('Inicialización de la webcam')
        self.webCam = cv2.VideoCapture(0)
        if self.webCam.isOpened():
            self.largoWebcam = self.webCam.get(3)
            self.anchoWebcam = self.webCam.get(4)
            print('Resolución:' + str(self.largoWebcam) + " X " + str(self.anchoWebcam))
        else:
            print('ERROR')

        while True:

            #De manera predeterminada no se hace ninguna detección.
            self.letraPredicha = False

            # Captura de la imagen en la variable Frame
            # La variable lecturaOK es igual a True si la función read() está operativa
            (self.lecturaOK, self.frame) = self.webCam.read()

            (self.grabbed, self.frame) = self.webCam.read()
            self.tsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            self.gris = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.contornos_canny = cv2.Canny(self.gris, 30, 200)

            self.contornos = cv2.findContours(self.contornos_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            for contorno in self.contornos:
                self.perimetro = cv2.arcLength(contorno, True)
                self.approx = cv2.approxPolyDP(contorno, 0.012 * self.perimetro, True)
                self.x, self.y, self.w, self.h = cv2.boundingRect(self.approx)

                #Se encuadra la zona de escritura en función de los parámetros de largo y ancho de la pizarra
                if len(self.approx) == 4 and self.h>self.zonaEscrituraAnchoMin and self.w>self.zonaEscrituraLargoMin and self.h<self.zonaEscrituraAnchoMax and self.w<self.zonaEscrituraLargoMax:

                    #Encuadre de la zona de escritura
                    self.area = cv2.contornoArea(contorno)
                    cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 3)

                    # Captura de la imagen a partir de la zona de escritura con un margen interior (padding) de 10
                    # píxeles para aislar solo la letra
                    self.letra = self.gris[self.y + 10:self.y + self.h - 10, self.x + 10:self.x + self.w - 10]

                    # Se detectan los contornos de la letra con la ayuda del algoritmo de Canny
                    self.cannyLetra = cv2.Canny(self.letra, 30, 200)
                    self.contornosLetra = cv2.findContornos(self.cannyLetra.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

                    # Si hay una letra d dibujada
                    if len(self.contornosLetra) > 5:

                        # Creación de una tabla para el almacenamiento de la imagen de la letra
                        self.captureAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                        # Se detecta el contorno más grande (Reverse = True)
                        self.cnt = sorted(self.contornosLetra, key=cv2.contornoArea, reverse=True)[0]

                        # Se guardan las coordenadas del rectángulo de delimtación de la letra
                        self.xc, self.yc, self.wc, self.hc = cv2.boundingRect(self.cnt)


                        for contornoLetra in self.contornosLetra:
                            self.area = cv2.contourArea(contorno)
                            if self.area > 1000:

                                # Se dibujan los contornos de la letra para una lectura mejor (Trazo de 10 px)
                                cv2.drawContours(self.captureAlphabetTMP, contornoLetra, -1, (255, 255, 255), 10)

                                # Se captura la letra y se guardan los valores de los píxeles de la zona capturada en una tabla
                                self.capturaLetra = np.zeros((400, 400), dtype=np.uint8)
                                self.capturaLetra = self.captureAlphabetTMP[self.yc:self.yc + self.hc, self.xc:self.xc + self.wc]

                                # Se pueden capturar sombras en la zona de escritura provocando errores de
                                # reconocimiento. Si se dectecta una sombra, una de las dimensiones de la tabla de captura es
                                # igual a cero porque no se ha detectado ningún contorno de letra
                                self.visualizacionLetraCapturada = True
                                if (self.capturaLetra.shape[0] == 0 or self.capturaLetra.shape[1] == 0):
                                    print("¡ERROR A CAUSA DE LAS SOMBRAS!: ")
                                    self.visualizacionLetraCapturada = False

                                #Si no es una sombra, se muestra la letra capturada en la pantalla
                                if self.visualizacionLetraCapturada:
                                    cv2.destroyWindow("ContornosLetra")
                                    cv2.imshow("ContornosLetra", self.capturaLetra)

                                    # Redimensionamiento de la imagen
                                    self.newImage = cv2.resize(self.captureLettre, (28, 28))
                                    self.newImage = np.array(self.newImage)
                                    self.newImage = self.newImage.astype('float32') / 255
                                    self.newImage.reshape(1, 28, 28, 1)

                                    # Realizatión de la predicción
                                    self.prediccion = self.cnn_model.predict(self.newImage.reshape(1, 28, 28,1))[0]
                                    self.prediccion = np.argmax(self.prediccion)

                                    # Se indica que se ha detectado una letra
                                    letraPredicha = True


                        if self.letraPredicha:

                            #Se desactiva la lectura de letras en voz alta
                            print('Desactivación de la lecture de letra ' + str(duracionDesactivacionLecturaDeLetra) + " segundos")
                            self.lectureActivee = False

                            #Se muestra el número de la letra predicho
                            #Se añade +1 porque la primera letra del alfabeto tiene valor 0 en nuestro sistema de predicción
                            #Entonces tiene el valor 1 en nuestra tabla de correspondencia
                            print("Detección:" + str(self.letraPredicha))
                            print("Predicción = " + str(self.prediccion))

                            #Lectura en voz alta de la letra predicha
                            if (self.letraPredicha and self.prediccion != 26):
                                self.engine.say('Leo la letra ' + str(self.letras[int(self.prediccion) + 1]))
                                self.engine.runAndWait()
                                letraPredicha = False

                            if (self.letraPredicha and self.prediccion == 26):
                                self.engine.say('No comprendo la letra escrita')
                                self.engine.runAndWait()
                                self.letraPredicha = False

                            #Pausa del proceso de lectura de la letra y luego llama a la funciónn para la reactivación de la
                            #lectura
                            self.timer = threading.Timer(self.duracionDesactivacionLecturaDeLetra, self.activacionLectura)
                            self.timer.start()


            # Visualización de la imagen capturada por la webcam
            cv2.imshow("IMAGEN", self.frame)
            cv2.imshow("HSV", self.tsv)
            cv2.imshow("GRIS", self.gris)
            cv2.imshow("CANNY", self.contornos_canny)

            # Condición de salida del bucle While
            # > Tecla Escape para salir
            self.key = cv2.waitKey(1)
            if self.key == 27:
                break

    def finalizar(self):
        #Se libera la webCam y se destruyen todas las ventanas
        self.webCam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def ejecutar():
        etapa3= Etapa3()
        etapa3.activacionLectura()
        etapa3.dimensiones()
        etapa3.inicioVoz()
        etapa3.eleccionVoz()
        etapa3.pruebaVoz()
        etapa3.aprendizaje()
        etapa3.inicioWebcam()
        etapa3.finalizar()


if __name__ == '__main__':
    Etapa3.ejecutar()
