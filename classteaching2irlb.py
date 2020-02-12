
#from sympy.interactive.printing import init_printing
import numpy as np
from scipy.optimize import minimize
#init_printing(use_unicode=False, wrap_line=True)
from RLIRL import *
from envs import *

np.set_printoptions(precision=2, suppress=True)
    
#select environment
envs = ["teethbrush", "mathematics", "random", "difflambda", "mix"]
#envs = ["difflambda"]
Rest = np.zeros((5,2,len(envs)))
nag = 2
envnum = -1 
for envname in envs:
    envnum += 1
    print(envname)

    LQ = []
    Lpol = []
    LR = []
    Ll = []
    LP = []
    #compute demonstrations for each learner
    for ll in range(0,nag):
        P,R,l,ns,na = environment(envname,ll)
        LP.append(P)
        #print("player " + str(ll))  
        #Find Policy
        Q = VI(P,R,ns,na,l)
        pol =  np.argmax(Q,axis=1)

        Ll.append(l)
        LQ.append(Q)
        Lpol.append(pol)
    All = set(list(range(0,ns)))
    Res = []
    for met in ['class0','class1','individual','augclass','approx']:
    #for met in ['augclass']:
        
        if met == 'approx':
            res = teach2diffstudents(LP[0],LP[1],Ll[0],Ll[1],LQ[0],LQ[1],R)
            Qapprox = np.reshape(res.x,(ns,na))
            polapprox = np.argmax(Qapprox,axis = 1)
        show = []
        if met == 'individual':
            pol = []
        elif met == 'class0':
            pol = Lpol[0]
        elif met == 'class1':
            pol = Lpol[1]
        elif met == 'augclass':
            Q = []
            pol = []
            common = list(np.nonzero(Lpol[0]==Lpol[1])[0])
        elif met =='approx':
            Q = Qapprox
            pol = polapprox

        LAs = []
        Ltl = []
        Lshow = []
        showt = set([])
        for ll in range(0,nag): 
            #Compute constraints    
            D = computconst(Lpol[ll],LP[ll],Ll[ll],ns,na)
            #Simplify Demonstration
            As,bs,tl = removeconstraints(D,na)
            LAs.append(As)
            Ltl.append(tl)
            z = np.ones((ns*(na-1)))
            z[tl] = 0
            show = np.nonzero(np.sum(z.reshape((ns,na-1)),axis=1))[0].tolist()        
            Lshow.append(show)

            showt = showt.union(Lshow[ll])
        showt = list(showt)
             
        avgeffort = 0
        avgJerr = 0
        avgpolerr = 0
        
        for ll in range(0,nag):

            show = showt
            eff = len(showt)/nag
            if met == 'augclass':
                Scommon = set(common)
                show = Scommon.union(set(Lshow[ll]))
                pol = Lpol[ll]
                eff = len(common)/nag + len(show)-len(common)
            elif met == 'individual':
                showt = All.difference(Ltl[ll])
                pol = Lpol[ll]
                show = Lshow[ll]
                eff = len(show)

            
            D = computconst(pol,LP[ll],Ll[ll],ns,na,show)
            #Simplify Demonstration
            As = D 
                    
            #Demonstrate and Learn
            r = lpIRL(As,Ll[ll],LP[ll])
            r = r[:,np.newaxis]
            #Verify policies
            Ql = VI(LP[ll],r,ns,na,Ll[ll])
            poll = np.argmax(Ql,axis=1)
            Q2 = VI(LP[ll],R,ns,na,Ll[ll],poll)
            aux2 = np.sum(np.diag((Q2[:,poll[:]]-LQ[ll][:,Lpol[ll][:]])))
            avgJerr += aux2
            avgeffort += eff

            avgpolerr += len(np.nonzero(Lpol[ll]-poll)[0])/len(pol)
            if True:
                print("computconst pol",pol)
                print("\n met", met, "player " + str(ll))
                print("demo pol    ",pol)
                print("learned pol ", poll)
                print("opt pol     ", Lpol[ll])
                print("show ", show, "avg effort ", avgeffort)    
                print("J error ",aux2, "avg J error ",avgJerr)

        Res.append([avgeffort/ns,avgJerr/2])
    Rest[:,:,envnum] += np.array(Res)

    print(np.round(np.array(Res),3))

