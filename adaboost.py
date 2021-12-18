from numpy.core.fromnumeric import resize
import clasificador_debil as cd
import numpy as np
import math
import utils

def entrenar(X, Y, T, A):
    clasificadores_debiles = []
    alphas = []
    #print(len(X))
    #num de pruebas aleatorio
    D=np.empty(len(X))
    D.fill(1/len(X))
    for i in range(T):
        minimo=-1,()
        for k in range(A):
            clasificador=cd.generar_clasificador_debil(28*28)
            error= cd.obtener_error2(clasificador,X,Y,D)
            if error<minimo[0] or minimo[0]==-1:
                minimo=(error,clasificador)

        clasificadores_debiles.append(minimo[1])
        errorTemp=(1-error)/error

        if errorTemp<=0:
            alpha=0.5
        else:
            alpha= (1/2)*math.log(((1-error)/error),2)
        alphas.append(alpha)

        Z=np.sum(D)
        for j in range(len(X)):
            Dcopia=D.copy()
            if cd.aplicar_clasificador_debil(minimo[1],X[j]):
                r=1
            else:
                r=-1
            Dcopia[j]=D[j]*np.e**(-alpha*Y[j]*r)
            #print(Dcopia[j],":::", D[j])
        
        D=Dcopia/Z
        print(D)
        
    return (clasificadores_debiles, alphas)



def test(X,Y,CF):
    CD=[]
    h=0
    contador=0
    for i in range(len(X)):
        h=0.0
        for j in range(len(CF[0])):
            
            aplicar=cd.aplicar_clasificador_debil(CF[0][j],X[i])
            if aplicar:
                resul=1
            else:
                resul=-1
            h+=CF[1][j]*resul
        CD.append(np.sign(h))
        CD[i]=int(CD[i])

        
    
    for k in range(len(CD)):
        if CD[k]==Y[k]:
            contador=contador+1
    print((contador/len(CD))*100,"%")
    return np.sign(h)






    # D[i]=1/X.len 
    # T --> indica el num de clasificadores debiles
    # X=X[0:1000] --> coger de 60000 mil imagenes 1000
    # 

    # Datos
    
    #imagenes_X = [[3,234,50], [1,89,100], [245,130,134]]
    #etiquetas_Y = [1, -1, 1]

    # imagenes_X = X[0:num]
    # etiquetas_Y = Y[0:num]

    # # Obtenemos un clasificador debil
    # CD = cd.generar_clasificador_debil(28*28)

    # # Aplicamos el clasificador a una imagen
    # res = cd.aplicar_clasificador_debil(CD, imagenes_X[0])

    # # Calculamos el error 
    # error = cd.obtener_error(CD, imagenes_X, etiquetas_Y, D)

    # ##########################

    # print("COMPLETAR ENTRENAMIENTO")
    #H devuelve 1 o -1
    