import numpy as np
import random
from RLIRL import *
from envs import *
#np.set_printoptions(precision=4, suppress=True)
envname = "chain"

# Example code for the paper:
#Lopes, M., Melo, F., & Montesano, L. (2009, September).
#Active learning for reward estimation in inverse reinforcement learning.
#In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 31-46).
#Springer, Berlin, Heidelberg.
## !!!this is not the original code!!!
##

# Task to teach
P,RO,l,ns,na = environment(envname,0)
RO = RO * 0
RO[5] = 1
QO = VI(P,RO,ns,na,l)
polO =  np.argmax(QO,axis=1)

# List of possible rewards
hypR = np.eye(ns)
# ns possible rewards where each one corresponds to
# reward of 1 in a single state
# in this case the true reward is in the hypothesis space

hypQ = np.zeros((ns,na,ns))
ic = 0
for ii in hypR:
    # compute policies corresponding to each R in hypR
    # P is assumed to be known
    Q = VI(P,ii[:,np.newaxis],ns,na,l)
    hypQ[:,:,ic]=Q2OptAct(Q)
    ic+=1

D = []
# intial likelihood of each hyp
likhyp = np.ones((ns,))/ns

# active phase
for ii in range(0,ns):    
    #weight votes
    p = np.sum((hypQ*likhyp),axis=2)
    e = np.nan_to_num(-p*np.log(p),0)
    we = np.sum(e,axis=1)    
    #print(np.nonzero(we == np.max(we))[0])
    ss = random.choice(np.nonzero(we == np.max(we))[0])
    a = polO[ss]
    # add element (ss,a) to the demonstration
    D.append((ss,a))
    likhyp *= np.exp(5*hypQ[ss,a,:]-1)
    likhyp = likhyp/np.sum(likhyp)
    #print(ss,a,likhyp)
    # if the likelihood of a given hypothesis is bigger than 50% than
    # it is probably it is the best one
    if np.max(likhyp>0.5):
        print("Final Dataset",D)
        break

rid = np.argmax(likhyp)
print("Identified reward",hypR[:,rid])
print("True reward",RO.transpose())
