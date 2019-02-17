import numpy as np
import time
import math
import string
import random
class dpmf:
	def __init__(self,soadd,ratadd,unum,inum,ratdim,socdim,steps,alpha,alpha1,beta,gamma, train_rate):
		self.unum=unum
		self.inum=inum
		self.train_rate = train_rate
		self.U=self.loadsocial(soadd)
		self.R,self.T=self.loadrating(ratadd)
		self.result=self.dp(ratdim,socdim,steps,alpha,alpha1,beta,gamma)


	def loadsocial(self,stri):
		f=open(stri,'r')
		temp=f.readline().strip().split(' ')
		u=int(temp[0])
		k=int(temp[1])
		Cu=np.zeros((self.unum,k))
		count=0
		for l in f:
			temp=l.strip().split(' ')
			i=int(temp[0])
			for j in range(k):
				Cu[i,j]=float(temp[j+1])
			count+=1
		f.close()
		return Cu

	def loadrating(self,stri):
		D=[]
  		with open(stri,'r') as handle:
    	        	for eachline in handle:
        	                sample=eachline.split('\n')[0].split('\t')
              	         	user=string.atoi(sample[0])
           		        item=string.atoi(sample[1])
                        	rating=string.atoi(sample[2])
                   		D.append([user,item,rating])
		random.shuffle(D)
		pos=int(len(D)*self.train_rate)
		return D[:pos],D[pos:]
	def maermse(self,P,Q,V):
	        m=0.0
		mae=0.0
	        rmse=0.0
	        n=0
	        for l in self.T:
	                i=l[0]
	                j=l[1]
	                m=math.fabs(l[2]-sum(P[i,:]*Q[:,j])-sum(self.U[i,:]*V[:,j]))
	                mae+=m
			rmse+=(m**2)
			n+=1
	        MAE=mae/n
	        RMSE=math.sqrt(rmse/n)
	        return MAE,RMSE


	def dp(self,ratdim,socdim,steps,alpha,alpha1,beta,gamma):
		starttime=time.clock()
		P=(np.random.rand(self.unum,ratdim))/math.sqrt(ratdim+socdim)
		Q=(np.random.rand(ratdim,self.inum))/math.sqrt(ratdim+socdim)

		V=(np.random.rand(socdim,self.inum))/math.sqrt(socdim+ratdim)
		perror=99999
		cerror=9999
		n=len(self.R)
		for step in xrange(steps):
			sum=0.0
			for l in self.R:
				i=l[0]
				j=l[1]
				eij=l[2]-np.dot(P[i,:],Q[:,j])-np.dot(self.U[i,:],V[:,j])
				sum+=(eij**2)
				P[i,:]=P[i,:]-alpha*(-eij*Q[:,j]+beta*P[i,:])
				Q[:,j]=Q[:,j]-alpha*(-eij*P[i,:]+beta*Q[:,j])
				V[:,j]=V[:,j]-alpha1*(-eij*self.U[i,:]+gamma*V[:,j])
			perror=cerror
			cerror=sum/n
			if(abs(perror-cerror)<0.0001):
                        	break

			alpha*=0.93
			alpha1*=0.93
			print 'step:',step
			MAE,RMSE=self.maermse(P,Q,V)
			print 'MAE,RMSE:',MAE,RMSE
			endtime=time.clock()
			print 'time:',endtime-starttime
		return 0
if __name__=='__main__':
	socialaddress='../data/embedding/douban.deepwalk.emb'
	ratingaddress='../data/ratings/douban.ratings'
	usernumber=3023
	itemnumber=6972
	ratingdimensions=10
	socialdimensions=40
	steps=100
	alpha=0.02
	alpha1=0.002
	beta=0.05
	gamma=0.5
	recommed=dpmf(socialaddress,ratingaddress,usernumber,itemnumber,ratingdimensions,socialdimensions,steps,alpha,alpha1,beta,gamma)
