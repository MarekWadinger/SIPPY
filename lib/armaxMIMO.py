# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 2017

@author: Giuseppe Armenise
"""

from functionset import *

def ARMAX_MISO_id(y,u,na,nb,nc,theta,max_iterations):
    nb=np.array(nb)
    theta=np.array(theta)
    u=1.*np.atleast_2d(u)
    ylength=y.size
    ystd,y=rescale(y)
    [udim,ulength]=u.shape
    eps=np.zeros(y.size)
    Reached_max=False
    if nb.size!=udim:
        print "Error! nb must be a matrix, whose dimensions must be equal to yxu"
        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    elif theta.size!=udim:
        print "Error! theta matrix must have yxu dimensions"
        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    else:
        nbth=nb+theta
        Ustd=np.zeros(udim)
        for j in range(udim):
            Ustd[j],u[j]=rescale(u[j])
        val=max(na,np.max(nbth),nc)
        N=ylength-val
        phi=np.zeros(na+np.sum(nb[:])+nc)
        PHI=np.zeros((N,na+np.sum(nb[:])+nc))
        for k in range(N):
            phi[0:na]=-y[k+val-1::-1][0:na] 
            for nb_i in range(udim):
                phi[na+np.sum(nb[0:nb_i]):na+np.sum(nb[0:nb_i+1])]=u[nb_i,:][val+k-1::-1][theta[nb_i]:nb[nb_i]+theta[nb_i]]
            PHI[k,:]=phi
        Vn=np.inf
        Vn_old=np.inf
        THETA=np.zeros(na+np.sum(nb[:])+nc)
        ID_THETA=np.identity(THETA.size)
        lambdak=0.5 
        iterations=0
        while (Vn_old>Vn or iterations==0) and iterations<max_iterations:
            THETA_old=THETA
            Vn_old=Vn
            iterations=iterations+1
            for i in range(N):
                PHI[i,na+np.sum(nb[:]):na+np.sum(nb[:])+nc]=eps[val+i-1::-1][0:nc]
            THETA=np.dot(np.linalg.pinv(PHI),y[val::])
            Vn=(np.linalg.norm(y[val::]-np.dot(PHI,THETA),2)**2)/(2*N)
            THETA_new=THETA
            lambdak=0.5
            while Vn>Vn_old:
                THETA=np.dot(ID_THETA*lambdak,THETA_new) + np.dot(ID_THETA*(1-lambdak),THETA_old)
                Vn=(np.linalg.norm(y[val::]-np.dot(PHI,THETA),2)**2)/(2*N)
                if lambdak<np.finfo(np.float32).eps:
                    THETA=THETA_old
                    Vn=Vn_old
                lambdak=lambdak/2.
            eps[val::]=y[val::]-np.dot(PHI,THETA)
        if iterations>=max_iterations:
            print "Warning! Reached maximum iterations"
            Reached_max=True
        DEN=np.zeros((udim,val+1))
        NUMH=np.zeros((1,val+1))
        NUMH[0,0]=1.
        NUMH[0,1:nc+1]=THETA[na+np.sum(nb[:])::]
        DEN[:,0]=np.ones(udim)
        NUM=np.zeros((udim,val))
        for k in range(udim):
            THETA[na+np.sum(nb[0:k]):na+np.sum(nb[0:k+1])]=THETA[na+np.sum(nb[0:k]):na+np.sum(nb[0:k+1])]*ystd/Ustd[k]
            NUM[k,theta[k]:theta[k]+nb[k]]=THETA[na+np.sum(nb[0:k]):na+np.sum(nb[0:k+1])]
            DEN[k,1:na+1]=THETA[0:na]
        return DEN,NUM,NUMH,Vn,Reached_max

#MIMO function
def ARMAX_MIMO_id(y,u,na,nb,nc,theta,tsample=1.,max_iterations=100):
    na=np.array(na)
    nb=np.array(nb)
    nc=np.array(nc)
    theta=np.array(theta)
    [ydim,ylength]=y.shape
    [udim,ulength]=u.shape
    [th1,th2]=theta.shape
    if na.size!=ydim:
        print "Error! na must be a vector, whose length must be equal to y dimension"
        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nc.size!=ydim:
        print "Error! nc must be a vector, whose length must be equal to y dimension"
        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nb[:,0].size!=ydim:
        print "Error! nb must be a matrix, whose dimensions must be equal to yxu"
        return 0.,0.,0.,0.,0.,0.,np.inf
    elif th1!=ydim:
        print "Error! theta matrix must have yxu dimensions"
        return 0.,0.,0.,0.,0.,0.,np.inf
    elif (np.issubdtype((np.sum(nb)+np.sum(na)+np.sum(nc)+np.sum(theta)),int) and np.min(nb)>=0 and np.min(na)>=0 and np.min(nc)>=0 and np.min(theta)>=0)==False:
        print "Error! na, nb, nc, theta must contain only positive integer elements"
        return 0.,0.,0.,0.,0.,0.,np.inf
    else:
        Vn_tot=0. 
        NUMERATOR=[]
        DENOMINATOR=[]
        DENOMINATOR_H=[]
        NUMERATOR_H=[]
        for i in range(ydim):
            DEN,NUM,NUMH,Vn,Reached_max=ARMAX_MISO_id(y[i,:],u,na[i],nb[i,:],nc[i],theta[i,:],max_iterations)
            if Reached_max==True:
                print "at ", (i+1),"° output"
                print "-------------------------------------"
            DENOMINATOR.append(DEN.tolist())
            NUMERATOR.append(NUM.tolist())
            NUMERATOR_H.append(NUMH.tolist())
            DENOMINATOR_H.append([DEN.tolist()[0]])
            Vn_tot=Vn+Vn_tot
        G=cnt.tf(NUMERATOR,DENOMINATOR,tsample)
        H=cnt.tf(NUMERATOR_H,DENOMINATOR_H,tsample)
        return DENOMINATOR,NUMERATOR,DENOMINATOR_H,NUMERATOR_H,G,H,Vn_tot

#creating object ARMAX MIMO model
class ARMAX_MIMO_model:
    def __init__(self,na,nb,nc,theta,ts,NUMERATOR,DENOMINATOR,NUMERATOR_H,DENOMINATOR_H,G,H,Vn):
        self.na=na
        self.nb=nb
        self.nc=nc
        self.theta=theta
        self.ts=ts
        self.NUMERATOR=NUMERATOR
        self.DENOMINATOR=DENOMINATOR
        self.NUMERATOR_H=NUMERATOR_H
        self.DENOMINATOR_H=DENOMINATOR_H
        self.G=G
        self.H=H
        self.Vn=Vn