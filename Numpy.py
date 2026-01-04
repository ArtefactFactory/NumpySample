# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 09:18:02 2026

@author: Vincent
"""

import numpy as np

#Création d'arrays

#1.Vecteur
A = np.array([1,2,3])
print(A)
type(A)

#2.Matrice
 
B = np.array([[1,2,3],[4,5,6]])
print(B)
 
C = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(C)
 
#3. Les attributs les plus utiles des objets arrays
 
#L'attribut shape
 
B.shape
C.shape
type(C.shape)
 
#L'attribut size
B.size
 
#L'attribut ndim
A.ndim
B.ndim
C.ndim
 
#4. Les constructeurs alternatifs de ndarray
 
Z = np.zeros((2,3))
print(Z)

O = np.ones((3,5))
print(O)

F = np.full((2,4), 5)
print(F)

L = np.linspace(-5,5,10)
print(L)
L.size

M = np.arange(-5,5,2)
print(M)
M.size

#5.Création d'arrays avec random

G = np.random.rand(5)
print(G)

T = np.random.rand(400,5000)
print(T)

P = np.random.randn(50)
print(P)

W = np.random.randint(10,size=5)
print(W)

N = np.random.randint(1000, size=(2000,3000))
print(N)
N.shape

R = np.random.randint(-10,10, size=(10,10))
print(R)

#Le calcul matriciel avec Numpy

#1. Addition et soustraction

A = np.array([[-2,-3,0], [1,2,3]])
print(A)

n_lin = A.shape[0]
n_col = A.shape[1]

B = np.ones((n_lin, n_col))
print(B)

C = A + B
print(C)

#2. Multiplication d'une matrice par un scalaire

coeff = 0.1
D = np.random.randint(-10,10,size=(3,3))
print(D)

E = coeff*D
print(E)

#3. Produit terme à terme entre deux matrices

F = np.array([[1,2,3],[4,5,6]])
G = np.array([[1,2,3],[4,5,6]])
H = F*G
print(F)
print()
print(G)
print()
print(H)

#4. Produit matriciel

I = np.random.randint(-2,20,size=(5,3))
J = np.random.randint(-10,0,size=(3,4))

K = np.dot(I,J)
print(K)

K.shape

#Le boolean indexing

#Le boolean indexing est un syntaxe particulière permettant d'accéder 
#à certains éléments d'une matrice répondant à une condition que 
#l'on spécifie entre crochets.
#si M est une matrice, alors M[M>0] renverra une matrice dans laquelle 
#il n'y a que des éléments strictement positifs.

#Indexing

A = np.array([1,2,3,4])
for i in range(len(A)):
    print(A[i])
    
B = np.array([[1,2,3],[4,5,6]])
print(B)  

B[0,2]
B[0,-1]

#Slicing

D = np.random.randint(-100,100,size=(4,4))
print(D) 

E = D[:,-1]
print(E)

F = D[1:3,:]
print(F) 

G = D[1:3,1:3]
print(G)

H = D[:,0:3:2]
print(H)

#Boolean indexing

I = np.random.randn(2,3)
print(I)

print(I>0)

J = I[I>0]
print(J)

I[I>0] = 0
print(I)



