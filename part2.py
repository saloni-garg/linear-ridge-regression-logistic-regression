import matplotlib.pyplot as plt
import math
import numpy as np
import random
#---------------------------------------
x_data =list()	
y_data = list()
output=list()	#output vector
#------------------------------------------
#reading file

data_mat = list()

with open("credit.txt") as f:
	for line in f:
		my_line = line.split(",")
		temp=list()
		temp.append(1)
		temp.append(float(my_line[0]))
		temp.append(float(my_line[1]))
		x_data.append(float(my_line[0]))
		y_data.append(float(my_line[1]))
		output.append(float(my_line[2]))	#storing the output 	
		data_mat.append(temp)                 #storing data in data_mat

#------------------------------------------
#Plotting data                                                    #--------------------
ps_x=list()													#-------To show plow uncomment plt.show below
ng_x=list()
ps_y=list()
ng_y=list()
for i in range(100):
	if(output[i]==0):
		ng_x.append(x_data[i])
		ng_y.append(y_data[i])
	if(output[i]==1):
		ps_x.append(x_data[i])
		ps_y.append(y_data[i])

plt.scatter(ps_x,ps_y,color=['brown'])
plt.scatter(ng_x,ng_y,color=['green'])	

#plt.show()

#-------------------------------------------
# sigmoid function
def sigmoid(x,w):
	x=np.transpose(x)
	y= np.matmul(x,w)

	y=(math.exp(-y))+1
	return 1/y


#function which returns the vector of all the outputs produced by the sigmoid function with given weight
def calc(x,w):
	result=list()
	for i in range(0,len(x)):
		vv =sigmoid(x[i],w)
		
		if(vv>0.5):
			vv=1
		else:
			vv=0	
		result.append(vv)
	return result	


#compares two list
def compare(l1,l2):
	vv=[i for i, j in zip(l1, l2) if i == j]
	if(len(vv)==len(l1)):
		return 1
	else:
		return 0

#count the number of mismatches
def err(f,y):
	err=0.0
	for i in range(len(f)):
		if f[i]!=y[i]:
			err+=1
	return err	

#------------------------------------------------------------------------------------
degree=2							#-------------------------Change the degree here|
#------------------------------------------------------------------------------------

len_weights = ((degree+1)*(degree+2))/2
w=list()
for i in range(len_weights):
	p = random.uniform(-0.1, 0.1)
	w.append(p)


#--------------------------------------------------

#Increasing the degree higher
#return new data set with (deg+1)*(deg+2)/2 terms
def highdegree(x,degree):
	n=((degree+1)*(degree+2))/2
	highdegree_x=[]
	for i in x:
		
		l=list()
		for m in range(n):
			l.append(0.0)
		m=0
		for x_deg in range(0, degree+1):
			for y_deg in range(0, degree+1):
				if x_deg==0 and y_deg==0:
					l[m]=1.0
					m=m+1
				elif x_deg+y_deg<degree+1:
					l[m]=(i[1]**x_deg)*(i[2]**y_deg)
					m=m+1
		highdegree_x.append(l)
	return highdegree_x

#------------------------------------------------------	
lmbda=0.01                                  # Taking lambda parameter to be 0.1 make it zero if you don't want to use regularized one.

data_mat = highdegree(data_mat,degree)
i=0
while(i!=100):
	i=i+1
	x_trans = np.transpose(data_mat)
	pp = np.subtract(calc(data_mat,w), output)
	st = np.matmul(x_trans,pp)
	eye = np.identity(100)
	ft = np.matmul(np.matmul(x_trans,eye),data_mat)
	ft = ft+ lmbda*np.identity(len_weights)
	ft = np.linalg.inv(ft) 
	t = np.matmul(ft,st)
	prev_w = w
	w = np.subtract(np.transpose(w),t)

	if(compare(prev_w,w)==1):
		break

out = calc(data_mat,w)
print ('Mismatches when degree is '+str(degree)+'='+str(err(out,output)))
#---------------------------------------------------------------------------






