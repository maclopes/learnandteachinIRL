# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:12:54 2020

@author: mlopes
"""

#test environments
import numpy as np

#Model
def environment(env,ll):
  
  if env=="demo1":

    na = 2
    Pa = np.array([[1-p1,p1,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    Pb = np.array([[0,0,1,0,0],  [0,0,0,0,1],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    R = np.array([[0],[0],[1],[0],[2]])
    ns = 5

    Pa = np.array([[1-p1,p1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,1,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    Pb = np.array([[0,0,1,0,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,0,1],[0,0,p1,1-p1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    R = np.array([[0],[0],[1],[0],[2],[0],[1]])
    ns = 7
    
  elif env=="difflambda":
    na = 2
    Pa = np.array([[0,1.,0,0,0],[0,0,0,1.,0],[0,0,1.,0,0],[0,0,0,1.,0],[0,0,0,0,1]])
    Pb = np.array([[0,0,1.,0,0],  [0,0,0,0,1.],[0,0,1.,0,0],[0,0,0,1.,0],[0,0,0,0,1]])
    R = np.array([[0],[0],[1],[0],[2]])
    ns = 5
    l = 0.9 * (1-ll) + 0.01 * ll
    P = (Pa,Pb)
    
  elif env=="mathematics":

    p = 0.1
    p1 = p * (1-ll) + 1 * ll
    l = 0.9
    
    na = 2
    #mathematics  
    ns = 8
    #write
    R = np.array([[0],[0],[0],[0],[1],[0],[0],[1]])
    Pa = np.zeros((ns,ns))
    Pa[0,1] = 1-p1
    Pa[0,3] = p1
    Pa[1,2] = 1
    Pa[2,2] = 1
    Pa[3,4] = 1
    Pa[4,4] = 1
    Pa[5,6] = 1
    Pa[6,7] = 1
    Pa[7,7] = 1

    #writetransport
    Pb = np.eye(ns)
    Pb[0,0] = 0
    Pb[0,5] = 1
    P = (Pa,Pb)
    

  elif env=="random":
    l = 0.9
    ns = 6
    na = 3
    Pa = np.array([[1,2,1,2+ll,4,5],[1,2,1,2+10*ll,4,5],[1,2,1,2+10*ll,4,5],[1,2,1,2+ll,4,5],[1,2,1,2+ll,4,5],[7,2,1,2+ll,4,5]])
    Pb = 1-Pa
    Pc = Pa**2+Pb
    Pa = Pa/Pa.sum(axis=1)[:,np.newaxis]
    Pb = Pb/Pb.sum(axis=1)[:,np.newaxis]
    Pc = Pc/Pc.sum(axis=1)[:,np.newaxis]
    R = np.array([[0],[0],[-1],[0],[0],[1]])
    P = (Pa,Pb,Pc)

  elif env=="mix":
    l = 0.9
    ns = 3
    na = 3
    if ll==0:
#      Pa = np.array([[0,1.,0],[0,.9,.1],[0,0,1.]])
#      Pb = np.array([[.9,.1,0],[0,0,1.],[0,0,1.]])
#      Pa = np.array([[0,1.,0],[0,1,0],[0,0,1.]])
#      Pb = np.array([[1.,0,0],[0,1,0],[0,.9,.1]])
      Pa = np.array([[0,1.,0],[0,1,0],[0,1,0]])
      Pb = np.array([[0,0,1.],[0,1,0],[0,1,0]])
      Pc = np.array([[1.,0,0],[0,1,0],[0,1,0]])
    else:
#      Pa = np.array([[.9,.1,0],[0,0,1.],[0,0,1.]])
#      Pb = np.array([[0,1.,0],[0,.9,.1],[0,0,1.]])
#      Pa = np.array([[1,0,0],[0,1,0],[0,1.,0]])
#      Pb = np.array([[.1,.9,0],[0,1,0],[0,0,1.]])
      Pa = np.array([[1.,0,0],[0,1,0],[0,1,0]])
      Pb = np.array([[0,0,1.],[0,1,0],[0,1,0]])
      Pc = np.array([[0,1.,0],[0,1,0],[0,1,0]])
    R = np.array([[0],[1.],[0]])
    P = (Pa,Pb,Pc)

  elif env=="teethbrush":

    p = 0.5
    p1 = p * (1-ll) + 1 * ll
    l = 0.92    
    #teethbrush
    ns = 10
    na = 4
    R = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[1]])
    Pa = np.zeros((ns,ns))
    Pa[0,1] = 1
    Pa[1,0] = 1
    Pa[2,3] = 1
    Pa[3,2] = 1
    if(ll==0):
      Pa[4,4] = 1
      Pa[5,5] = 1
      Pa[6,6] = 1
      Pa[7,7] = 1
    else:
      Pa[4,5] = 1
      Pa[5,4] = 1
      Pa[6,7] = 1
      Pa[7,6] = 1

    Pa[8,8] = 1
    Pa[9,9] = 1

    Pb = np.zeros((ns,ns))
    Pb[0,4] = 1
    Pb[4,0] = 1
    if(ll==0):
      Pb[1,1] = 1
      Pb[5,5] = 1
      Pb[2,2] = 1
      Pb[6,6] = 1
    else:
      Pb[1,5] = 1
      Pb[5,1] = 1
      Pb[2,6] = 1
      Pb[6,2] = 1
    Pb[3,7] = 1
    Pb[7,3] = 1
    Pb[8,8] = 1
    Pb[9,9] = 1

    Pc = np.eye(ns)
    Pc[1,2] = 1
    Pc[1,1] = 0
    Pc[5,6] = 1
    Pc[5,5] = 0

    Pd = np.eye(ns)
    if(ll==0):
      Pd[6,8] = 0
    else:
      Pd[6,8] = 1
      Pd[6,6] = 0
    Pd[7,9] = 1
    Pd[7,7] = 0

    P = (Pa,Pb,Pc,Pd)

  return P,R,l,ns,na