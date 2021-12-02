import matplotlib.pyplot as plt
import numpy as np

def mostrar_imagen(imagen):
    plt.figure()
    plt.imshow(imagen)
    plt.show()

def adaptar_conjuntos(mnist_X, mnist_Y):
    #print("COMPLETAR ADAPTACION")
    X=mnist_X.reshape(60000, 28*28)
    return (X,mnist_Y)

def plot_arrays(X, Y, title):
    plt.title(title)
    plt.plot(X, Y)
    plt.show()


def adaptarY(mnist_Y, clase):
    Y=np.full(len(mnist_Y),0)
    for i in range(len(Y)):
        if mnist_Y[i]==clase:
            Y[i]=1
        else:
            Y[i]=-1
    return Y