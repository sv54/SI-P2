# Importamos las librerias que necesitaremos
import numpy as np
import matplotlib.pyplot as plt
import utils
import adaboost
import os
import clasificador_debil as cd


def main():

    
    # for f in os.listdir("/Users/Serg2/Source/repos/SI-P2/SI-P2"):
	#     print(f)
    
    # Cargamos la base de datos
    npzfile = np.load("/Users/Serg2/Source/repos/SI-P2/SI-P2/mnist.npz") ##QUITAR LUEGO
    #npzfile = np.load("mnist.npz")
    mnist_X = npzfile['x']
    mnist_Y = npzfile['y']
    #utils.mostrar_imagen(mnist_X[0])
    #print(mnist_Y[0])
    num=300
    incr=300

    (X, Y) = utils.adaptar_conjuntos(mnist_X, mnist_Y)
    X = X[0:num]
    Y = Y[0:num]

    Y=utils.adaptarY(Y,0)    
    # Lanzar Adaboost
    T = 500
    A = 5
    #print(len(X))
    c_f = adaboost.entrenar(X, Y, T, A)
    
    print("Entrenamiento acabado", len(c_f[0]))
    
    num=num+incr
    (X,Y)=utils.adaptar_conjuntos(mnist_X,mnist_Y)
    X=X[num:num+incr]
    Y=Y[num:num+incr]
    Y=utils.adaptarY(Y,0)
    adaboost.test(X,Y,c_f)
        

if __name__ == "__main__":
    main()


# Mostrar una imagen y su etiqueta


# Adaptar los conjuntos X e Y a AdaBoost



# Analisis y resultados de las pruebas realizadas
#T = [0, 100, 200, 300, 400]      # Numero de clasificadores 
#resultados = [0, 20, 35, 56, 68] # Resultados obtenidos de clasificacion
#utils.plot_arrays(T, resultados, "Porcentajes con valores de T")
    