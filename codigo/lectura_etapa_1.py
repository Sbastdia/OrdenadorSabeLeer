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

class Etapa1:

    def __init__(self):
        #dimensiones de la pizarra
        self.zonaEscrituraLargoMin = 540
        self.zonaEscrituraLargoMax = 590
        self.zonaEscrituraAnchoMin = 300
        self.zonaEscrituraAnchoMax = 340

    def inicializacion(self):
        #Inicialización de la webcam
        print('Inicialización de la webcam')
        self.webCam = cv2.VideoCapture(0)
        if self.webCam.isOpened():
            self.largoWebcam = self.webCam.get(3)
            self.anchoWebcam = self.webCam.get(4)
            print('Resolución:' + str(self.largoWebcam) + " X " + str(self.anchoWebcam))
        else:
            print('ERROR')

    def captura(self):
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
                    cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 3)

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

    def finalizacion(self):
        #Se libera la webCam y se destruyen las ventanas
        self.webCam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def ejecutar():
        etapa1 = Etapa1()
        etapa1.inicializacion()
        etapa1.captura()
        etapa1.finalizacion()

if __name__ == '__main__':
    Etapa1.ejecutar()