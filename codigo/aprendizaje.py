#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 13 - El ordenador saber leer
#
# Modulos necesarios:
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


from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from mnist import MNIST
import numpy as np
from matplotlib import pyplot as plt

class Aprendizaje:

    def __init__(self):
        #Carga de las imágenes
        self.emnist_data = MNIST(path='datas\\', return_type='numpy')
        self.emnist_data.select_emnist('letters')
        self.Imagenes, self.Etiquetas = self.emnist_data.load_training()


        print("Cantidad de imágenes ="+str(len(self.Imagenes)))
        print("Cantidad de etiquetas ="+str(len(self.Etiquetas)))


    def conversion(self):
        #Conversión de las imágenes y etiquetas en tabla numpy

        self.Imagenes = np.asarray(self.Imagenes)
        self.Etiquetas = np.asarray(self.Etiquetas)

    def dimension(self):
        #Dimensión de las imégenes de trabajo y de aprendizaje
        self.largoImagen = 28
        self.anchoImagen = 28

    def visualizacion(self):
        #Las imágenes están en la forma de una tabla de 124800 líneas y 784 columnas
        #Las transformamos en una tabla que contiene 124800 líneas que contiene una tabla de 28*28 columnas
        print("Transformación de las tablas de imágenes...")
        self.Imagenes = self.Imagenes.reshape(124800, self.anchoImagen, self.largoImagen)
        self.Etiquetas= self.Etiquetas.reshape(124800, 1)

        print("Visualización de la imagen N.° 70000...")

        plt.imshow(self.Imagenes[70000])
        plt.show()

        print(self.Etiquetas[70000])

        #En informática, los índices de las listas deben empezar por cero...")
        self.Etiquetas = self.Etiquetas-1


        print("Etiqueta de la imagen N.° 70000...")
        print(self.Etiquetas[70000])


    def aprendizaje(self):
        #Creación de los conjutnos de aprendizaje y de prueba
        self.imagenes_aprendizaje, self.imagenes_validacion, self.etiquetas_aprendizaje, self.etiquetas_validacion = train_test_split(self.Imagenes, self.Etiquetas, test_size=0.25, random_state=42)

        #Adición de un tercer valor a nuestras tablas de imágenes para que puedan ser utilizadas por la red neuronal, especialmente el parámetro input_shape de la función Conv2D
        self.imagenes_aprendizaje = self.imagenes_aprendizaje.reshape(self.imagenes_aprendizaje.shape[0], self.anchoImagen, self.largoImagen, 1)
        print(self.imagenes_aprendizaje.shape)

        self.imagenes_validacion = self.imagenes_validacion.reshape(self.imagenes_validacion.shape[0], self.anchoImagen, self.largoImagen, 1)

        #Creación de una variable que sirve de imagen de trabajo a la red neuronal
        self.imagenTrabajo = (self.anchoImagen, self.largoImagen, 1)

    def adaptacion(self):
        #Adaptación a la escala
        self.imagenes_aprendizaje = self.imagenes_aprendizaje.astype('float32')/255
        self.imagenes_validacion = self.imagenes_validacion.astype('float32')/255

    def codificacion(self):
        # Creación de las categorías en un sistema de codificación One-Hot
        self.cantidad_de_clases = 26
        self.etiquetas_aprendizaje = keras.utils.to_categorical(self.etiquetas_aprendizaje, self.cantidad_de_clases)
        self.etiquetas_validacion = keras.utils.to_categorical(self.etiquetas_validacion, self.cantidad_de_clases)

    def red(self):
        # Red neuronal convolucional
        # 32 filtros de dimensiones 3x3 con una función de activación de tipo RELU
        # El filtro tiene en la entrada la imagen de trabajo
        self.redCNN = Sequential()
        self.redCNN.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=self.imagenTrabajo))

        #Una segunda capa de 64 filtros de dimensión 3x3
        self.redCNN.add(Conv2D(64, (3, 3), activation='relu'))

        #Una función de pooling
        self.redCNN.add(MaxPooling2D(pool_size=(2, 2)))
        self.redCNN.add(Dropout(0.25))

        #Un aplanado
        self.redCNN.add(Flatten())

        #La red neuronal con 128 neuronas en la entrada
        #una función de activación de tipo ReLU
        self.redCNN.add(Dense(128, activation='relu'))
        self.redCNN.add(Dropout(0.5))

        #Una última capa de tipo softmax
        self.redCNN.add(Dense(self.cantidad_de_clases, activation='softmax'))

    def compilacion(self):
        #Compilación del modelo
        self.redCNN.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

    def entrenamiento(self):
        # Aprendizaje con una fase de validación
        # en los conjuntos de prueba
        self.batch_size = 128
        self.epochs = 10

        self.redCNN.fit(self.imagenes_aprendizaje, self.etiquetas_aprendizaje,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(self.imagenes_validacion, self.tiquetas_validacion))

    def guardado(self):
        # Guardado del modelo
        self.redCNN.save('modelo/modelo_caso_practicov2.h5')

    def evaluacion(self):
        # Evaluación de la precisión del modelo
        self.score = self.redCNN.evaluate(self.imagenes_validacion, self.etiquetas_validacion, verbose=0)
        print('Precisión en los datos datos de validación:', self.score[1])

    @staticmethod
    def ejecutar():
        #Creamos un objeto de la clase Aprendizaje
        aprendizaje = Aprendizaje()
        aprendizaje.conversion()
        aprendizaje.dimension()
        aprendizaje.visualizacion()
        aprendizaje.aprendizaje()
        aprendizaje.adaptacion()
        aprendizaje.codificacion()
        aprendizaje.red()
        aprendizaje.compilacion()
        aprendizaje.entrenamiento()
        aprendizaje.guardado()
        aprendizaje.evaluacion()

if __name__ == '__main__':
    Aprendizaje.ejecutar()