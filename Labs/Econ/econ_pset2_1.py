#Econ Week 2 Pset: Exercise Set 1

#EXERCISE 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, solve

A = [[0.6, 0.1, -0.3],
     [0.5, -0.4, 0.2],
     [1.0, -0.2, 1.1]]

b = [[12],
     [10],
     [-1]]

A, b = map(np.asarray, (A, b))

evs = eigvals(A)
rho = max(abs(evs))
print(rho)

'''
The spectral radius condition holds if the largest absolute value of the eigenvalues \
of A is less than 1. It holds because the largest absolute eigenvalue is 0.965.
'''

#Computation with fixed point iteration
def fixedpoint(A, b):
    eps = 1e-9
    x = [[1],
         [1],
         [1]]

    diff = np.linalg.norm(A @ x + b - x)
    while (diff > eps):
        x = A @ x + b
        diff = np.linalg.norm(A @ x + b - x)

    return x

print(fixedpoint(A, b))

#Computation with matrix algebra
'''
We rewrite the equation A@x - b = x as A_t@x = b_t where b_t = -b.
'''
A_t = [[-0.4, 0.1, -0.3],
     [0.5, -1.4, 0.2],
     [1.0, -0.2, 0.1]]

b_t =[[-12],
     [-10],
     [1]]

print(np.linalg.solve(A_t, b_t))

'''
We see that both methods yield the same solution.
'''

#EXERCISE 2
'''
See LATEX
'''

#EXERCISE 3
def reswage(beta, w1, w2, w3, p1, p2, p3):
    c_vals = np.linspace(1, 2, 100)
    eps = 1e-9
    x = 1
    wvec = list([])
    for c in c_vals:
        diff = 5
        while(diff > eps):
            x_new =  c*(1-beta) + beta*(max(w1,x)*p1 + max(w2,x)*p2 + max(w3,x)*p3)
            diff = abs(x_new - x)
            x = x_new.copy()
            del(x_new)
        wvec.append(x)
    plt.plot(wvec)
    plt.show()
    return wvec

print(reswage(0.96, 0.5, 1.0, 1.5, 0.2, 0.4, 0.4))
