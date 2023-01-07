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

class Etapa2:

    def __init__(self):
        #dimensiones de la pizarra
        self.zonaEscrituraLargoMin = 540
        self.zonaEscrituraLargoMax = 590
        self.zonaEscrituraAnchoMin = 300
        self.zonaEscrituraAnchoMax = 340

    def iniciar(self):
        #Inicialización de la webcam
        print('Inicialización de la webcam')
        self.webCam = cv2.VideoCapture(0)
        if self.webCam.isOpened():
            self.largoWebcam = self.webCam.get(3)
            self.anchoWebcam = self.webCam.get(4)
            print('Resolución:' + str(self.largoWebcam) + " X " + str(self.anchoWebcam))
        else:
            print('ERROR')

    def leer(self):
        # Captura de la imagen en la variable Frame
        while True:

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

                    # Se detectan los contornos de la letra con la ayuda del algoritmo Canny
                    self.cannyLetra = cv2.Canny(self.letra, 30, 200)
                    self.contornosLetra = cv2.findContornos(self.cannyLetra.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

                    # Si hay una letra d dibujada
                    if len(self.contornosLetra) > 5:

                        # Creación de una tabla para el almacenamiento de la imagen de la letra
                        self.capturaAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                        # Se detecta el contorno más grande (Reverse = True)
                        self.cnt = sorted(self.contoursLettre, key=cv2.contourArea, reverse=True)[0]

                        # Se guardan las coordenadas del rectángulo de delimitación de la letra
                        self.xc, self.yc, self.wc, self.hc = cv2.boundingRect(self.cnt)


                        for contornoLetra in self.contornosLetra:
                            self.area = cv2.contornoArea(contorno)
                            if self.area > 1000:

                                # Se dibujan los contornos de la letra para una lectura mejor (Trazo de 10 px)
                                cv2.drawContours(self.capturaAlphabetTMP, contornoLetra, -1, (255, 255, 255), 10)

                                # Se captura la letra y se guardan los valores de los píxeles de la zona capturada en una tabla
                                self.capturaLetra = np.zeros((400, 400), dtype=np.uint8)
                                self.capturaLetra = self.capturaAlphabetTMP[self.yc:self.yc + self.hc, self.xc:self.xc + self.wc]


                                #Se pueden capturar sombras en la zona de escritura provocando errores de
                                #reconocimiento. Si se dectecta una sombra, una de las dimensiones de la tabla de captura es
                                #igual a cero porque no se ha detectado ningún contorno de letra
                                self.visualizaciónLetraCapturada = True
                                if (self.capturaLetra.shape[0] == 0 or self.capturaLetra.shape[1] == 0):
                                    print("¡ERROR A CAUSA DE LAS SOMBRAS!: ")
                                    self.visualizaciónLetraCapturada = False

                                #Si no es una sombra, se muestra la letra capturada en la pantalla
                                if self.visualizaciónLetraCapturada:
                                    cv2.destroyWindow("ContornosLetra")
                                    cv2.imshow("ContornosLetra", self.capturaLetra)

                                    # Redimensionamiento de la imagen
                                    self.newImage = cv2.resize(self.capturaLetra, (28, 28))
                                    self.newImage = np.array(self.newImage)
                                    self.newImage = self.newImage.astype('float32') / 255
                                    self.newImage.reshape(1, 28, 28, 1)



            # Visualización de la imagen capturada por la webcam
            cv2.imshow("IMAGEN", self.frame)
            cv2.imshow("HSV", self.tsv)
            cv2.imshow("GRIS", self.gris)
            cv2.imshow("CANNY", self.contornos_canny)

            # Condición de salida del bucle While
            # > Tecla Escape para salir
            self.key = cv2.waitKey(1)
            if self.ey == 27:
                break

    def finalizar(self):
        #Se libera la webcam y se destruyen todas las ventanas
        self.webCam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def ejecutar():
        etapa2= Etapa2()
        etapa2.iniciar()
        etapa2.leer()
        etapa2.finalizar()

if __name__ == '__main__':
    Etapa2.ejecutar()