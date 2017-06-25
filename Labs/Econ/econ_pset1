# Import libraries
import numpy as np
import scipy
import scipy.optimize as opt
import matplotlib.pyplot as plt

#Exercise 5.1

# Household Parameters
nvec = np.array([1.0,1.0,0.2])
L = sum(nvec)
beta_annual = 0.96 #annual discount factor
beta = beta_annual  ** 20
sigma = 3 #sigma = scalar >=1, coefficient of relative risk aversion

# Firms Parameters
alpha = 0.35 #alpha represents percentage going to capital. 1-alpha represents percentage going to labor
A = 1.0 #A is total factor productivity
delta_annual = 0.05 #delta is depreciation rate
delta = 1 - ((1 - delta_annual) ** 20) #delta is depreciation over entire period (60 years)
params = A, alpha, delta

def qn1(bvec, *args):
    b2, b3 = bvec
    alpha, beta, sigma, A, delta = args

    w = (1-alpha)*A*(((b2+b3)/2.2)**alpha)
    r = (alpha)*A*((2.2/(b2+b3))**(1-alpha)) - delta
    c529 = w + (1+r)*b2 - b3
    c530 = (1+r)*b3 + 0.2*w
    uprime529 = 1/(c529**(sigma))
    uprime530 = 1/(c530**(sigma))
    Eulerr1 = 1/((w-b2)**sigma) - beta*(1+r)*uprime529
    Eulerr2 = 1/((w+(1+r)*b2-b3)**sigma) - beta*(1+r)*uprime530
    optarray= np.array([Eulerr1, Eulerr2])

    return optarray

def column(matrix, i):
    return np.array([row[i] for row in matrix]).reshape(len(matrix),1)

def get_r(K,L,params):
    A, alpha, delta = params
    r = alpha * A * ((L/K) ** (1-alpha)) - delta
    return r

def get_w(K,L,params):
    A, alpha, delta = params
    w = (1-alpha) * A * ((K/L)**alpha)
    return w

# Make initial guess for solution of savings values. Note that these
# two guesses must be feasible and not violate c_t > 0 for all t
b2_init = 0.02
b3_init = 0.06
b_init = np.array([b2_init, b3_init])
b_args = (alpha, beta, sigma, A, delta)
b_result = opt.root(qn1, b_init, args=(b_args))
print(b_result)
print("b2 is: ", b_result.x[0])
print("b3 is: ", b_result.x[1])

#r bar
print("r bar is: ", get_r(sum(b_result.x),sum(nvec),params))

#w bar
print("w bar is: ", get_w(sum(b_result.x),sum(nvec),params))

#c1, c2, c3
print("c1 is: ", get_w(sum(b_result.x),sum(nvec),params) - b_result.x[0])
print("c2 is: ", get_w(sum(b_result.x),sum(nvec),params) +
      (1+get_r(sum(b_result.x),sum(nvec),params))*b_result.x[0]- b_result.x[1])
print("c3 is: ", 0.2*get_w(sum(b_result.x),sum(nvec),params) +
      (1+get_r(sum(b_result.x),sum(nvec),params))*b_result.x[1])

#Exercise 5.2
print("All the steady-state values increase. The intuition is that as beta "
      "increases, households discount future consumption streams less and save "
      "more. The increase in savings stimulates investment and economic growth, "
      "which increases consumption and wages.")
''

#Exercise 5.3
#STEADY STATE VALUES
b2_bar = 0.0193127352392
b3_bar = 0.0584115908789
K_bar = b2_bar + b3_bar

#GUESS TIME PATH FOR K
T = 31
bmatsim = np.zeros((T,3))
bmatsim[0][0] = 0.8*b2_bar
bmatsim[0][1] = 1.1*b3_bar
bmatsim[0][2] = bmatsim[0][0] + bmatsim[0][1]
print(bmatsim)

def linearpath(K1,Kbar,bmat):
    bmat[0][2] = K1
    if K1>Kbar:
        for i in range(1,len(bmat)-1):
            bmat[i][2] = bmat[i-1][2] - (K1-Kbar)/(len(bmat)-1)
    else:
        for i in range(1,len(bmat)-1):
            bmat[i][2] = bmat[i-1][2] + (K1-Kbar)/(len(bmat)-1)
    bmat[len(bmat)-1][2] = Kbar

    return bmat

#bmatreal = Matrix with only first row and column of K's
bmatreal = linearpath(0.8*b2_bar + 1.1*b3_bar, K_bar, bmatsim)
print(bmatreal)

bmatcopy = np.array(bmatreal, copy=True)

#rwmat are the r's and w's implied by Knoprime
def get_rwmat(L,params,bmat):
    A, alpha, delta = params
    rwmat = np.zeros((len(bmat),2)) #31x2 matrix with r,w
    for i in range(0,len(rwmat)):
        rwmat[i][0] = get_r(bmat[i][2],L,params)
        rwmat[i][1] = get_w(bmat[i][2],L,params)

    return rwmat

rwmat = get_rwmat(L,params,bmatreal)
print(rwmat)


def getEulErr_one(b3, *args):
    r1, r2, w1, w2, beta, sigma = args
    Eulone = ((1+r1)*0.0193127352392 + w1 - b3)**(-sigma) - beta*(1+r2)*(((1+r2)*b3 + 0.2*w2)**(-sigma))

    return Eulone

def getEulErr_two(bvec, *args):
    b2, b3 = bvec
    r2, r3, w1, w2, w3, beta, sigma = args
    Eulerr1 = (w1-b2)**(-sigma) - beta*(1+r2)*(((1+r2)*b2 + w2 - b3)**(-sigma))
    Eulerr2 = ((1+r2)*b2 + w2 - b3)**(-sigma) - beta*(1+r3)*(((1+r3)*b3 + 0.2*w3)**(-sigma))
    Eularray = np.array([Eulerr1, Eulerr2])

    return Eularray

def getroot1(rwmat, beta, sigma, getEulErr):
    r1 = rwmat[0][0]
    r2 = rwmat[1][0]
    w1 = rwmat[0][1]
    w2 = rwmat[1][1]
    b_args_1 = r1, r2, w1, w2, beta, sigma
    root1 = opt.root(getEulErr, 0.05, args=(b_args_1)).x
    return root1

print(getroot1(rwmat, beta, sigma, getEulErr_one)) #b_3_2

#Bmatsim it takes in is matrix with Knoprime
def get_bmatsim(rwmat, bmatsim, b_3_2):
    b2_init = 0.02
    b3_init = 0.06
    b_init = np.array([b2_init, b3_init])

    bmatsim[1][1] = b_3_2
    for i in range(1, len(bmatsim)-1):
        b_args_2 = rwmat[i][0], rwmat[i+1][0], rwmat[i-1][1], rwmat[i][1], rwmat[i+1][1], beta, sigma
        bmatsim[i][0] = opt.root(getEulErr_two, b_init, args=(b_args_2)).x[0]
        bmatsim[i+1][1] = opt.root(getEulErr_two, b_init, args=(b_args_2)).x[1]
    bmatsim[len(bmatsim)-1][0] = b2_bar
    for i in range(0, len(bmatsim)):
        bmatsim[i][2] = bmatsim[i][0] + bmatsim[i][1]

    return bmatsim

#bmatsimfinal = Matrix with savings distribution (b-values) backed out from
#r's and w's that were in turn from Knoprime
bmatsimfinal = get_bmatsim(rwmat,bmatsim,
                           getroot1(rwmat, beta, sigma, getEulErr_one))
print(bmatsimfinal)
Knoprime = column(bmatcopy,2)
Knoprime_init = np.array(Knoprime, copy=True)
Kprime = column(bmatsimfinal, 2)
eps = 1e-9
difference = np.linalg.norm(Kprime - Knoprime)
xi = 0.5

while(difference > eps):
    bmatiter = np.zeros((T,2))
    bmatiter[0][0] = 0.8*b2_bar
    bmatiter[0][1] = 1.1*b3_bar
    Knoprime = xi*Kprime + (1-xi)*Knoprime
    bmatiter = np.c_[bmatiter, Knoprime]
    rwmatiter = get_rwmat(L,params,bmatiter)
    b_3_2 = getroot1(rwmatiter, beta, sigma, getEulErr_one)
    bmatsimiter = get_bmatsim(rwmatiter,bmatiter,b_3_2) #matrix of b2, b3, and Kprime
    Kprime = column(bmatsimiter, 2)
    difference = np.linalg.norm(Kprime - Knoprime)

plt.plot(Kprime)
plt.plot(Knoprime_init)
plt.savefig("K_path.png")
print(bmatiter)
