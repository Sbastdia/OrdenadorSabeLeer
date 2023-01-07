from codigo.aprendizaje import Aprendizaje
from codigo.lectura_etapa_1 import Etapa1
from codigo.lectura_etapa_2 import Etapa2
from codigo.lectura_etapa_3 import Etapa3

if __name__ == '__main__':
    print("¿Qué desea ejecutar?")
    opcion = input("1. Aprendizaje\n2. Etapa 1\n3. Etapa 2\n4. Etapa 3\n")

    if opcion == "1":
        Aprendizaje().ejecutar()
    elif opcion == "2":
        Etapa1().ejecutar()
    elif opcion == "3":
        Etapa2().ejecutar()
    elif opcion == "4":
        Etapa3().ejecutar()
    else:
        print("Opción no válida")