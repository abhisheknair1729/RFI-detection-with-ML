#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def _matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print(e)
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization(R, P, Q, K, steps=500, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print(step)
        ij = numpy.where(R > 0)
        #print(type(ij[0]))
        #print(R>0)
        eij = R[ij] - numpy.matmul(P,Q)[ij]
        #print(eij.shape)
        ii = numpy.array(list(ij[0]))
        jj = numpy.array(list(ij[1]))
        #print(ii) 
        for k in range(K):
            P[ii][:,k] = P[ii][:,k] + alpha * (2 * numpy.multiply( eij, Q[k][jj] ) - beta * P[ii][:,k] )
            Q[k][jj] = Q[k][jj] + alpha * ( 2 * numpy.multiply( eij, P[ii][:,k] ) - beta * Q[k][jj] )
        
        e = 0;
        e = e + sum(pow(R[ij] - numpy.matmul(P,Q)[ij], 2))
        for k in range(K):
            e = e + sum((beta/2) * ( pow(P[ii][:,k],2) + pow(Q[k][jj],2) ))
        print(e)
        if e < 0.001:
            break
    return P, Q.T



###############################################################################

if __name__ == "__main__":
    R = numpy.zeros((256, 256), dtype = int)
    for i in range(256):
        for j in range(256):
            R[i][j] = (pow(i,j) + pow(j,i))%7


    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 64

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = _matrix_factorization(R, P, Q, K)
    nP1, nQ1 = matrix_factorization(R, P, Q, K)
    
    #print(nP)
    print(nP1)
    #assert(nP == nP1)
    #assert(nQ == nQ1)
