# Importamos las librerias que necesitaremos
import numpy as np
import matplotlib.pyplot as plt
import utils
import adaboost
import os
import clasificador_debil as cd


#Entrenar los 10 clasificadores fuertes
def entrenarCFs(X,mnist_Y):
    listaCFs=[0,1,2,3,4,5,6,7,8,9]
    num=1000 #numero de imagenes
    incr=100 
    A=50 #Num de intentos para cada clasificador
    T=100 #Num de clasificadores debiles
    for i in range(10):
        Y=utils.adaptarY(mnist_Y,i)
        listaCFs[i]=adaboost.entrenar(X[0:num],Y[0:num],T,A)

    return listaCFs


def testCFs(X,Y,listaCFs):
    print(len(listaCFs))
    listaImgRes=np.full((len(X),10),[0]) #[[-1,-1,1,-1,-1,-1,-1,-1,-1,-1][-1,-1,1,-1,-1,-1,-1,-1,-1,-1]]
    for i in range(len(listaCFs)):
        ResClase=adaboost.test(X,Y,listaCFs[i])
        for j in range(10):
            listaImgRes[j][i]=ResClase[j]

    
    resultado=0
    for i in range(len(X)):
        ultimo=-1
        for j in range(10):
            if(listaImgRes[i][j]==1):
                ultimo=j
                if (Y[i]==ultimo):
                    resultado=resultado+1
                
    

    print(resultado/len(X)*100)

    return 0


def main():

    # Cargamos la base de datos
    npzfile = np.load("/Users/Serg2/Source/repos/SI-P2/SI-P2/mnist.npz") ##Para cargar en PC, quitar luego
    #npzfile = np.load("mnist.npz")
    mnist_X = npzfile['x']
    mnist_Y = npzfile['y']
    #utils.mostrar_imagen(mnist_X[0])
    #print(mnist_Y[0])
    X,Y=utils.adaptar_conjuntos(mnist_X,mnist_Y)


    listaCF=entrenarCFs(X,mnist_Y)
    X=X[30000:31000]
    Y=Y[30000:31000]
    testCFs(X,Y,listaCF)






    

    # for i in range (10):
    #     num=1000
    #     incr=1000

    #     (X, Y) = utils.adaptar_conjuntos(mnist_X, mnist_Y)
    #     X = X[0:num]
    #     Y = Y[0:num]
    #     Y=utils.adaptarY(Y,2)

    #     # Lanzar Adaboost
    #     T = 10
    #     A = 10
    # #    print(Y)
    #     cf0 = adaboost.entrenar(X, Y, T, A)


    #     #print("Entrenamiento acabado", len(cf0[0]))
    #     for i in range(1):
    #         num=num+incr
    #         (X,Y)=utils.adaptar_conjuntos(mnist_X,mnist_Y)
    #         X=X[num:num+incr]
    #         Y=Y[num:num+incr]
    #         Y=utils.adaptarY(Y,2)
    #         adaboost.test(X,Y,cf0)


if __name__ == "__main__":
    main()






# Mostrar una imagen y su etiqueta


# Adaptar los conjuntos X e Y a AdaBoost



# Analisis y resultados de las pruebas realizadas
#T = [0, 100, 200, 300, 400]      # Numero de clasificadores 
#resultados = [0, 20, 35, 56, 68] # Resultados obtenidos de clasificacion
#utils.plot_arrays(T, resultados, "Porcentajes con valores de T")
    