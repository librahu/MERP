import numpy as np
import time
import math
import string
import random
import sys
class dpmf:
	def __init__(self,soadd,ratadd,unum,inum,ratdim,socdim,steps,alpha,alpha1,beta,gamma,k,gamma1,train_rate):
		self.unum=unum
		self.inum=inum
                self.train_rate = train_rate
		self.U=self.loadsocial(soadd,ratdim)
		self.R,self.T=self.loadrating(ratadd)
		self.result=self.dp(ratdim,socdim,steps,alpha,alpha1,beta,gamma,k,gamma1)


	def loadsocial(self,stri,ratdim):
		f=open(stri,'r')
		temp=f.readline().strip().split(' ')
		u=int(temp[0])
		k=int(temp[1])
#		if u!=self.unum:
#			print 'ERROR!'
		Cu=np.zeros((self.unum,k))
		count=0
		for l in f:
			temp=l.strip().split(' ')
			i=int(temp[0])
			for j in range(k):
				Cu[i,j]=float(temp[j+1])/math.sqrt(k)
			count+=1
		f.close()
		return Cu

	def loadrating(self,stri):
		D=[]
  		with open(stri,'r') as handle:
    	        	for eachline in handle:
        	                sample=eachline.split('\n')[0].split(' ')
              	         	user=string.atoi(sample[0])
           		        item=string.atoi(sample[1])
                        	rating=string.atoi(sample[2])
                   		D.append([user,item,rating])
		random.shuffle(D)
		pos=int(len(D)*self.train_rate)
		return D[:pos],D[pos:]
	def maermse(self,P,Q,E,V,b,u,bu,bi):
	        m=0.0
		mae=0.0
	        rmse=0.0
	        n=0
	        for l in self.T:
	                i=l[0]
	                j=l[1]
	                m=math.fabs(l[2]-sum(P[i,:]*Q[:,j])-np.dot(np.dot(self.U[i,:],E),V[:,j])-np.dot(self.U[i,:],b[i,:])-u-bu[i]-bi[j])
	                mae+=m
			rmse+=(m**2)
			n+=1
	        MAE=mae/n
	        RMSE=math.sqrt(rmse/n)
	        return MAE,RMSE


	def dp(self,ratdim,socdim,steps,alpha,alpha1,beta,gamma,k,gamma1):
		#f=open('rating_dimension_range_result','w')
		mae=[]
		rmse=[]
		starttime=time.clock()
		P=(np.random.rand(self.unum,ratdim))/math.sqrt(ratdim)
		Q=(np.random.rand(ratdim,self.inum))/math.sqrt(ratdim)
		E=(np.random.rand(socdim,k))/math.sqrt(socdim)
		V=(np.random.rand(k,self.inum))/math.sqrt(socdim)
		b=np.zeros((self.unum,socdim))
		bu=np.zeros(self.unum)
		bi=np.zeros(self.inum)
		u=3
		perror=99999
		cerror=9999
		n=len(self.R)
		for step in xrange(steps):
			sum=0.0
			for l in self.R:
				i=l[0]
				j=l[1]
				eij=l[2]-np.dot(P[i,:],Q[:,j])-np.dot(np.dot(self.U[i,:],E),V[:,j])-np.dot(self.U[i,:],b[i,:])-u-bu[i]-bi[j]
				sum+=(eij**2)
				bu[i]+=alpha*(eij-beta*bu[i])
				bi[j]+=alpha*(eij-beta*bi[j])
				P[i,:]=P[i,:]-alpha*(-eij*Q[:,j]+beta*P[i,:])
				Q[:,j]=Q[:,j]-alpha*(-eij*P[i,:]+beta*Q[:,j])
				temp1 = np.array([self.U[i,:]])
				temp2 = np.array([V[:,j]])
				b[i,:] = b[i,:] - alpha1*(-eij*self.U[i,:]+gamma*b[i,:])
				E=E-alpha1*(-eij*np.dot(temp1.T,temp2)+gamma1*E)
				#print temp1.shape
				#print temp2.shape
				#exit(0)
				temp3 = np.array([self.U[i,:]])
				V[:,j]=V[:,j]-alpha1*(-eij*np.dot(temp3,E)+gamma1*V[:,j])
			perror=cerror
			cerror=sum/n
			if(abs(perror-cerror)<0.0001):
                        	break

			alpha*=0.93
			alpha1*=0.93
			#print 'step:',step
			MAE,RMSE=self.maermse(P,Q,E,V,b,u,bu,bi)
			mae.append(MAE)
			rmse.append(RMSE)
		#	print 'MAE,RMSE:',MAE,RMSE
			endtime=time.clock()
		#	print 'time:',endtime-starttime
                print 'MAE: ',min(mae),'RMSE: ',min(rmse)
		#f.write('MAE:'+str(min(mae))+'RMSE:'+str(min(rmse))+'\n')
		return 0
if __name__=='__main__':
	socialaddress='../data/embedding/yelp.line.emb'
	ratingaddress='../data/ratings/yelp.ratings'
#	usernumber=14086
#	itemnumber=14038
	usernumber=9581
	itemnumber=14037
	ratingdimensions=10
	socialdimensions=64
	steps=100
	alpha=0.02
	alpha1=0.02
	beta=0.05
	gamma=0.05
	gamma1=0.05
	k=20
        train_rate = 0.2
        print train_rate
	recommed=dpmf(socialaddress,ratingaddress,usernumber,itemnumber,ratingdimensions,socialdimensions,steps,alpha,alpha1,beta,gamma,k,gamma1,train_rate)
