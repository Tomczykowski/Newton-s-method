import numpy as np
import matplotlib.pyplot as plt

def countGradient(x, y):
    gradient = np.array([2*(200*x**3 - 200*x*y + x - 1), 200*(y - x**2)])
    return gradient

def countHessian(x, y):
    hessian = np.array([[1200*x**2 - 400*y + 2, -400*x], [-400*x, 200]])
    hessian = np.linalg.inv(hessian)
    return hessian

def countD(x, y):
    hessian = countHessian(x, y)
    gradient = countGradient(x, y)
    d = hessian.dot(gradient)
    return d

def plotMaker(listX, listY):
    plt.plot(listX, listY, label='Function Value')
    plt.title("Newton")
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0)
    plt.legend()
    plt.grid(True)
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.xticks(np.arange(-5, 6, 1))
    plt.yticks(np.arange(-5, 6, 1))
    plt.savefig("plot.png")

def newton(x, y, b, epsilon, iCounter):
    i = 0
    listX = []
    listY = []
    d = countD(x, y)
    while i < iCounter and (abs(b * d[0]) > epsilon or abs(b * d[1]) > epsilon):
        listX.append(x)
        listY.append(y)
        i +=1
        d = countD(x, y)

        x = x - b * d[0]
        y = y - b * d[1]
    plotMaker(listX, listY)


newton(0, 3, 0.5, 10**-12, 500000)
