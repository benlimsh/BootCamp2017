import numpy as np
from matplotlib import pyplot as plt

#Problem 1
def var(n):
    arr = np.random.rand(n,n)
    arrmean = np.mean(arr, axis = 1)
    arrvar = np.var(arrmean)
    return arrvar
print(var(5))

def lln():
    vararr = []
    for n in range(100,1001,100):
        vararr.append(var(n))
    plt.plot(vararr)
    plt.show()
    return

#Problem 2
def trigplot():
    x = np.linspace(-2*np.pi, 2*np.pi, 100000)
    print(x)
    s = np.sin(x)
    c = np.cos(x)
    at = np.arctan(x)
    plt.plot(x,s)
    plt.plot(x,c)
    plt.plot(x,at)
    plt.show()
    return

#Problem 3
def f1plot():
    x1 = np.linspace(-2, 0.99, 100000)
    x2 = np.linspace(1.01, 6, 100000)
    y1 = np.true_divide(1,x1-1)
    y2 = np.true_divide(1,x2-1)
    plt.plot(x1, y1, 'm--', linewidth = 2)
    plt.plot(x2, y2, 'm--', linewidth = 2)
    plt.ylim(-6, 6)
    plt.show()
    return

#Problem 4
def f2plot():
    x = np.linspace(0, 2*np.pi)
    y1 = np.sin(x)
    y2 = np.sin(2*x)
    y3 = 2*np.sin(x)
    y4 = 2*np.sin(2*x)

    plt.subplot(221)
    plt.plot(x, y1, 'g-')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("sinx")

    plt.subplot(222)
    plt.plot(x, y2, 'r--')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("sin2x")

    plt.subplot(223)
    plt.plot(x, y3 ,'b--')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("2sinx")

    plt.subplot(224)
    plt.plot(x, y4, 'm:')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("2sin2x")

    plt.tight_layout()
    plt.show()

    return

#Problem 5
def fars():
    farsarr = np.load('FARS.npy')
    lon = farsarr[:,1]
    lat = farsarr[:,2]
    hours = farsarr[:,0]

    plt.subplot(121)
    plt.plot(lon, lat, "ko", markersize = 0.01)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.axis("equal")

    plt.subplot(122)
    plt.hist(hours, bins = 24,edgecolor='black', linewidth=1.2)
    plt.xlabel("hour")
    plt.tight_layout()
    plt.show()

    return

#Problem 6
def f3plot():
    x = np.linspace(-2*np.pi, 2*np.pi)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    Z = np.sin(X)*np.sin(Y)/(X*Y)*1.0

    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap = "viridis")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.xlim(-2*np.pi, 2*np.pi)

    plt.subplot(122)
    plt.contour(X, Y, Z, 20, cmap = "Spectral")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.xlim(-2*np.pi, 2*np.pi)

    plt.tight_layout()
    plt.show()
