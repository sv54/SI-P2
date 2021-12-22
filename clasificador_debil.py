import random
import numpy as np
from matplotlib.pyplot import cla
from numpy.lib.shape_base import apply_over_axes

# Dimension con la que vamos a trabajar. En nuestro caso 28*28
def generar_clasificador_debil(dimension_datos):
    #print("COMPLETAR GENERAR CLASIFICADOR DEBIL")
    pixel=random.randint(0,dimension_datos-1)
    umbral= random.randint(0,255)
    i=random.choice([True, False])
    if i:
        i=1
    else:
        i=-1
    return (pixel,umbral,i) # Devolvemos el clasificador debil generado

def aplicar_clasificador_debil(clasificador, imagen):
    
    pixel=clasificador[0]
    umbral=clasificador[1]
    i=clasificador[2]
    umbralImg=imagen[pixel]

    if i==1:
        if umbralImg>=umbral:
            return True
        else:
            return False
        
    if i==-1:
        if  umbralImg<umbral:
            return True
        else:
            return False


def obtener_error(clasificador, X, Y, D):
    error=0.0
    for k in range(len(X)):
        aplicar=aplicar_clasificador_debil(clasificador,X[k])
        
        if aplicar:
            if Y[k] != 1:
                error=error+D[k]
        else:
            if Y[k] == 1:
                error=error+D[k]
        #print("clasificador:",clasificador," error: ", error)
    return error
