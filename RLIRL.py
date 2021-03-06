# -*- coding: utf-8 -*-


#@title
from scipy.optimize import linprog
from scipy.optimize import minimize
import numpy as np

def Q2OptAct(Q):
  return (Q-np.max(Q,axis=1)[:,np.newaxis])==0

#remove redundant constraints using linear programming
def removeconstraints(A,na=1,b=[]):
  tl = []
  bounds  = (-1, 1)
  n = A.shape[0]
  if b==[]:
    b = np.zeros((n,1))
  if na>1:
    na = na - 1
  
  for ss in range(n//na):
    for aa in range(na):
      t = ss*na+aa

      c = A[t,:]
      At = -np.copy(A)
      bt = -np.copy(b)
      At = np.delete(At,tl+[t], axis=0)
      bt = np.delete(bt,tl+[t], axis=0)
      res = linprog(c, A_ub=At, b_ub=bt, bounds=(bounds), options={"presolve":False, "disp": False})
      x = res.x[:,np.newaxis]
#       print(A[t,:]@x)
#       print("At ", At)
#       print("bt ", bt)
#       print("x ", x)
#       print("c ", c)
      if (A[t,:]@x)>=0:
#         print("remove")
#         print(A[t,:]@x)
        tl.append(t)

  
  A = np.delete(A,(tl), axis=0)
  b = np.delete(b,(tl), axis=0)
      
  return A,b,tl


# compute constraints on given states show using policy pol
def computconst(pol,P,l,ns,na,show=[]):
  #Compute constraints  
  if show == []:
    show = list(range(0,ns))
  
  D = np.zeros((len(show)*(na-1),ns))
  ii = 0
  for ss in show:
    for aa in range(0,na):
      if aa==pol[ss]:
        continue
      D[ii,:] = (P[pol[ss]][ss,:]-P[aa][ss,:])
      ii += 1

  return D

# InverseReinforcementLearning from state-action constraints using linear programming
# We do not need feature counts and we solve the lp directly on V and reconstruct
# the reward R(X) afterwords.
def lpIRL(As, l, P,margin=.001):

  na = len(P)
  ns = P[0].shape[0]
  #D = np.hstack((As,np.ones((As.shape[0],1))))
  D = As
  bs = np.zeros((D.shape[0],1))+margin
  c = np.ones((1,D.shape[1]))
  #print(c.shape,As.shape,D.shape,bs.shape)
  res = linprog(c, A_ub=-D, b_ub=-bs, bounds=( (-1, 1) ), options={"presolve":False, "disp":True})
  #V = res.x[0:-1]+res.x[-1]
  V = res.x
 
  q = np.zeros((ns,na))
  for aa in range(0,na):
    q[:,aa] = l*P[aa]@V
  r = V-np.max(q,axis=1)

  print("x",res.x,"V",V,"r",r)
  print(-D@V,-bs)
  
  return r

def teach2diffstudents(P0,P1,l1,l2,Q0,Q1,R):
    V0 = np.max(Q0,axis=1)
    V1 = np.max(Q1,axis=1)
    ns = Q0.shape[0]
    na = Q0.shape[1]
    I = np.eye(ns)
    def teach2diffstudentsobjfunc(pol):
        # for now this minimizes the average error in V
        # maybe we would like to reduce the error in V(s_0)
        # what have we done in the previous cases????
        nx = np.reshape(pol, (ns,na))
        nx = np.argmax(np.exp(nx),axis=1)
        #nx = Q2pol(nx)
        Pol0=0
        Pol1=0
        for ii in range(0,na):
            dind = np.diag(nx==ii)*1
            #ding = np.diag(nx[:,ii])
            Pol0 += dind@P0[ii]
            Pol1 += dind@P1[ii]    
        mVerr = np.sum((np.linalg.pinv(I-l1*Pol0)@R-V0[:,None])**2+(np.linalg.pinv(I-l2*Pol1)@R-V1[:,None])**2)
    
        return mVerr
    
    res0 = minimize(teach2diffstudentsobjfunc, np.reshape(Q0,(ns*na,)), method='Nelder-Mead', options={'disp': False})
    res1 = minimize(teach2diffstudentsobjfunc, np.reshape(Q1,(ns*na,)), method='Nelder-Mead', options={'disp': False})
    res2 = minimize(teach2diffstudentsobjfunc, np.ones((ns*na,)), method='Nelder-Mead', options={'disp': False})
    #ret = dual_annealing(teach2diffstudents, bounds=([[0,100]]*(ns*na)), seed=1234)
    resv = (res0.fun,res1.fun,res2.fun)
    res = (res0,res1,res2)
    #print(resv)
    imin = np.argmin(np.array(resv))

    return res[imin]

if False:
    A = np.array([[-1, 0, 2], [0, -1, 3]])
    b = [0, 0]     
    As,bs,tl = removeconstraints(A,1,b)

    print(A)
    print("Final program")
    print(As)

    A = np.array([[-1, 2], [-1, 2], [-1, 3], [-1, 3]])
    As,bs,tl = removeconstraints(A)

    print(A)
    print("Final program")
    print(As)

"""Auxiliary functions"""

#@title
#Auxiliary functions

#Value Iteration
def VI(P,R,ns,na,l,pol = []):
  
  Q = np.zeros((ns,na))
  V = np.zeros((ns))
  
  err = 1
  while err>1e-12:
    for aa in range(0,na):
      Q[:,aa,np.newaxis] = R + l * P[aa]@V[:,np.newaxis]
    if pol==[]:
      nV = np.max(Q,axis=1)
    else:
      nV = np.diag(Q[:,pol])
    err = np.sum((nV-V)**2)
    V = nV
  return Q

#Compute trajectory using pol starting in state 0
def optpath(P,R,ns,na,pol, stochastic = 0):
  
  D = [0]
  r = 0
  s = 0
  while 1:
    p = P[pol[s]][s,:]
    s = np.nonzero(np.random.multinomial(1, p, size=1)[0])[0][0]
    #print("optpath s>", s)
    r = R[s]
    if r >= 1:
      if stochastic > 0:
        s = 0
        stochastic -= 1
        continue
      else:
        break
    D.append(s)
  
  return D
