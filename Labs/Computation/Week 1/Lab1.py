import numpy as np

#Problem 1
def prob(A,B):

    result = np.array([[0,0,0,0],
             [0,0,0,0]])

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    for r in result:
        print(r)

    return result

A = np.array([[3,-1,4],
    [1,5,-9]])

B = np.array([[2,6,-5,3],
    [5,-8,9,7],
    [9,-3,-2,-3]])

print(prob(A,B))

#Alternatively
def prob1():
    A = np.array([[3,-1,4],
        [1,5,-9]])

    B = np.array([[2,6,-5,3],
        [5,-8,9,7],
        [9,-3,-2,-3]])

    return A@B

print(prob1())

#Problem 2
from numpy import linalg as LA
def prob2():

    A=np.array([[3,1,4],[1,5,9],[-5,3,1]])
    Amat=np.matrix(A)
    result=-LA.matrix_power(Amat,3)+9*np.dot(Amat,Amat)-15*Amat
    return result

print(prob2())

#Problem 3
def prob3():
    A=np.triu(np.ones((7,7),dtype=np.int))
    ones=np.ones((7,7),dtype=np.int)
    triuones=np.triu(ones)
    B1=-1*np.tril(ones)
    np.fill_diagonal(triuones,0)
    B2=5*triuones
    B=B1+B2

    result=A@B@A
    result=result.astype(np.int64)
    return result

print(prob3())

#Problem 4
def prob4(arr):
    arr_copy=arr
    mask= arr_copy < 0
    arr_copy[mask] = 0
    return arr_copy

#Problem 5
def prob5():
    A=np.transpose(np.arange(6).reshape(3,2))
    B=3*np.tril(np.ones((3,3),dtype=np.int))
    C=np.diag([-2,-2,-2])
    A_t=np.transpose(A)
    zero_3x3=0*np.ones((3,3),dtype=np.int)
    zero_2x2=0*np.ones((2,2),dtype=np.int)
    zero_2x3=0*np.ones((2,3),dtype=np.int)
    zero_3x2=0*np.ones((3,2),dtype=np.int)

    I_3x3=np.eye(3)
    H1=np.hstack((zero_3x3,A_t,I_3x3))
    H2=np.hstack((A,zero_2x2,zero_2x3))
    H3=np.hstack((B,zero_3x2,C))
    block=np.vstack((H1,H2,H3))

    return block

print(prob5())

#Problem 6
def prob6(M):
    rowsum=M.sum(axis=1)
    M_norm=M/(rowsum*1.0)
    return M_norm

#Problem 7
def prob7():
    grid = np.load("/Users/benjaminlim/Downloads/grid.npy")
    maxh=np.max(grid[:,:-3]*grid[:,1:-2]*grid[:,2:-1]*grid[:,3:])
    print(maxh)
    maxv=np.max(grid[-3,:]*grid[1:-2,:]*grid[2:-1,:]*grid[3,:])
    print(maxv)
    maxdd=np.max(grid[:-3,:-3]*grid[1:-2,1:-2]*grid[2:-1,2:-1]*grid[3:,3:])
    maxud=np.max(grid[3:,:-3]*grid[2:-1,1:-2]*grid[1:-2,2:-1]*grid[:-3,3:])
    winner=max(maxh,maxv,maxdd,maxud)
    return winner

print(prob7())
