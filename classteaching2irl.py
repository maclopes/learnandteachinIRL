
#from sympy.interactive.printing import init_printing
import numpy as np
from scipy.optimize import minimize
#init_printing(use_unicode=False, wrap_line=True)
from RLIRL import *
from envs import *
"""Test Environments"""

#@title


"""Main loop"""



#@title

#select environment
envs = ["teethbrush", "mathematics", "random", "difflambda", "mix"]
#envs = ["mix"]
Rest = np.zeros((5,2,len(envs)))
               
envnum = -1 
for envname in envs:
    envnum += 1
    print(envname)
    
    LQ = []
    Lpol = []
    Lshow = []
    LR = []
    Ll = []
    #compute demonstrations for each learner
    P0,R,l,ns,na = environment(envname,0) 
    P1,R,l,ns,na = environment(envname,1) 
    
    for ll in range(0,2):
      P,R,l,ns,na = environment(envname,ll) 
      #print("player " + str(ll))  
      #Find Policy
      Q = VI(P,R,ns,na,l)
      pol =  np.argmax(Q,axis=1)
      #print(pol)
      #print(Q)
        
      Ll.append(l)
      LQ.append(Q)
      Lpol.append(pol)
      if envname == "random":
        show = np.unique(optpath(P,R,ns,na,pol,30))
        #Rest = np.zeros((5,3))
      else:
        show = optpath(P,R,ns,na,pol)
      show = list(range(0,ns))
      Lshow.append( show )
    
    Res = []
    for met in ['class0','class1','individual','augclass','approx']:
    #for met in ['approx']:
      avgeffort = 0
      avgJ = 0
      avgJerr = 0
      avgpolerr = 0
      if met == 'approx':
        res = teach2diffstudents(P0,P1,Ll[0],Ll[1],LQ[0],LQ[1],R)
        Qapprox = np.reshape(res.x,(ns,na))
        polapprox = np.argmax(Qapprox,axis = 1)
        #polapprox = np.array([1,0])
        showapprox = optpath(P,R,ns,na,polapprox)
        #showapprox = [0]
        #print("showapprox",showapprox)
    
      if met == 'augclass':
        common = list(set(Lshow[0]) & set(Lshow[1]))
        Lshow2 = [list(set(Lshow[0])-set(common)),list(set(Lshow[1])-set(common)),common]
        effort = len(Lpol[0]) + len(Lpol[1]) - 2*len(common)
        for ii in common:
          if Lpol[0][ii]==Lpol[1][ii]:
            effort += 1
          else:
            effort += 2
            Lshow2[0].append(ii)
            Lshow2[1].append(ii)
            Lshow2[2] = list( (set(Lshow2[2])-set([ii])))      
      
      for ll in [0,1]:
        #print("\n met", met, "player " + str(ll))  
        P,R,l,ns,na = environment(envname,ll) 
        
        if met == 'individual':
          Q = LQ[ll]
          pol = Lpol[ll]
          show = Lshow[ll]
          avgeffort += len(show)/ns
        elif met == 'class0':
          Q = LQ[0]
          pol = Lpol[0]
          show = Lshow[0]
          avgeffort += len(show)/ns
        elif met == 'class1':
          Q = LQ[1]
          pol = Lpol[1]
          show = Lshow[1]
          avgeffort += len(show)/ns
        elif met == 'augclass':
          Q = LQ[ll]
          pol = Lpol[ll]
          show = Lshow2[ll].copy()
          avgeffort += (2*len(Lshow2[ll])+len(Lshow2[2]))/ns
          show.extend(Lshow2[2])
        elif met =='approx':
          Q = Qapprox
          pol = polapprox
          show = showapprox
          avgeffort += len(show)/ns
    
        #Compute constraints    
        D = computconst(pol,P,l,ns,na,Q,show)
        As = D
        #Simplify Demonstration
        As,bs,tl = removeconstraints(As,na)    
        #print("demo ", tl)
    
        #Demonstrate and Learn
        r = lpIRL(As,l,P)
        r = r[:,np.newaxis]
        #print("r ", r)
        #Verify policies
        Ql = VI(P,r,ns,na,l)
        poll = np.argmax(Ql,axis=1)
        Q2 = VI(P,R,ns,na,l,poll)
        aux1 =  (Q2[0,poll[0]]-LQ[ll][0,Lpol[ll][0]])
       
        avgJ += aux1
        aux2 = np.sum(np.diag((Q2[:,poll[:]]-LQ[ll][:,Lpol[ll][:]])))
        avgJerr += aux2
        
        avgpolerr += len(np.nonzero(Lpol[ll]-poll)[0])/len(pol)
##        print("\n met", met, "player " + str(ll)) 
##        print("\n show ", show)
##        print("effort ", len(show)/ns)    
##        print("pol error   ", len(np.nonzero(Lpol[ll]-poll)[0])/len(pol))
##        print("opt pol     ", Lpol[ll])
##        print("learned pol ", poll)
##        print("J error ",aux1)
##        print("avg J error ",aux2)
    
      Res.append([avgeffort/2,avgJerr/2])
    Rest[:,:,envnum] += np.array(Res)
    
    print("taught policy ", pol)
    print("avgeffort ", avgeffort/2)
    print("avgJ ", avgJ/2)
    print("avg J error > ", avgJerr/2 )
    print("avgpole(rr ", avgpolerr/2)
    print(np.round(np.array(Res),3))

# res = teach2diffstudents(P0,P1,Ll[0],Ll[1],LQ[0]+LQ[1],LQ[0]+LQ[1],R)
# Qapprox = np.reshape(res.x,(ns,na))
# print(Qapprox)
# polapprox = np.argmax(Qapprox,axis = 1)
# print(polapprox)
# print(pol)
# D = computconst(pol,P,l,ns,na,Qapprox,show)
# As = D
# #Simplify Demonstration
# #     print("remove constraints")
# #As,bs,tl = removeconstraints(As,na)    
# #print("demo ", tl)

# #Demonstrate and Learn
# r = lpIRL(As,l,P)
# r = r[:,np.newaxis]
# print("r ", r)
# #Verify policies
# Ql = VI(P,r,ns,na,l)
# poll = np.argmax(Ql,axis=1)
# print("Ql",Ql)
# print("show ", show)
# print("effort ", len(show)/ns)    
# print("pol error   ", len(np.nonzero(Lpol[ll]-poll)[0])/len(pol))
# print("opt pol     ", Lpol[ll])
# print("learned pol ", poll)
# print(r)
