from numpy.core.fromnumeric import resize
import clasificador_debil as cd
import numpy as np
import math

def entrenar(X, Y, T, A):
    clasificadores_debiles = []
    alphas = []
    num = 100


    #print(D)
    A=10 #num de pruebas aleatorio
    for i in range(T):
        D=np.empty(num)
        D.fill(1/num)
        minimo=-1,()
        #error=0
        for k in range(A):
            clasificador=cd.generar_clasificador_debil(28*28)
            #print(clasificador)
            #error= cd.obtener_error(clasificador,X,Y,D)
            error= cd.obtener_error2(clasificador,X,Y,D)
            #print(error)
            if error<minimo[0] or minimo[0]==-1:
                minimo=(error,clasificador)
        #print(D)
        clasificadores_debiles.append(minimo[1])
        errorTemp=(1-error)/error
        #print(errorTemp)
        if errorTemp<=0:
            alpha=0.5
        else:
            alpha= (1/2)*math.log(((1-error)/error),2)
        #print (alpha)
        alphas.append(alpha)
        Z=sum(D)
        Dcopia=D
        for j in range(len(D)):
            if cd.aplicar_clasificador_debil(minimo[1],X[j]):
                r=1
            else:
                r=-1
            Dcopia[j]=D[j]*np.e**(-alpha*Y[j]*r)
        D=Dcopia/Z

    return (clasificadores_debiles, alphas)



def test(X,Y,c_f):
    c_d=[]
    #print(c_f)
    for i in range(len(X)):
        h=0
        for j in range(len(c_f[0])):
            #print(c_f[0][j])
            aplicar=cd.aplicar_clasificador_debil(c_f[0][j],X[i])
            if aplicar:
                resul=1
            else:
                resul=-1
        h+=c_f[1][j]*resul
        c_d.append(np.sign(h))
        
        c_d[i]=int(c_d[i])
        
    contador=0
    for k in range(len(c_d)):
        
        if c_d[k]==Y[k]:
            contador=contador+1
    print(c_d)
    print(contador)
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
    # c_d = cd.generar_clasificador_debil(28*28)

    # # Aplicamos el clasificador a una imagen
    # res = cd.aplicar_clasificador_debil(c_d, imagenes_X[0])

    # # Calculamos el error 
    # error = cd.obtener_error(c_d, imagenes_X, etiquetas_Y, D)

    # ##########################

    # print("COMPLETAR ENTRENAMIENTO")
    #H devuelve 1 o -1
    