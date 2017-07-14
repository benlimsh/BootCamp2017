import numpy as np

def newton(x, eps, fprime, fprimeprime):

    iterations = 0
    diff = 5
    while diff > eps and iterations < 1000:
        oldx = np.copy(x)
        x = x - fprime(x)/fprimeprime(x)
        diff = np.linalg.norm(x - oldx)/np.linalg.norm(oldx)
        iterations = iterations + 1
        del(oldx)

    if diff > eps:
        print("Function did not converge")
    else:
        print("Function converged. x = ", x)

    return x

def fprime(x):
    return 6*x**5 + 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1

def fprimeprime(x):
    return 5*6*x**4 + 4*5*x**3 + 3*4*x**2 + 2*3*x**1 + 2

newton(2, 1e-3, fprime, fprimeprime)
