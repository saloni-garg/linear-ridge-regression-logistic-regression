#------------------------------------------
#imports
#import statistics
import numpy
import math
from numpy.linalg import inv
import random
import matplotlib.pyplot as plt

#---------------------------------------------
data =list()	#Data Matrix

output=list()	#output vector

#----------------------------------------------------------------
#Function that returns the weights using analytical approach
def mylinridgereg(X, Y, lambd):
	identity = numpy.identity(len(X[0]))
	tt = lambd * numpy.array(identity)
	term1 = numpy.matmul(numpy.transpose(X),X)
	term1 = term1 + tt
	term1 = inv(term1)
	
	term2 = numpy.matmul(numpy.transpose(X),Y)
	return numpy.matmul(term1,term2)

#------------------------------------------------------------------
#Function that returns the predicted output using the input data and weights
def mylinridgeregeval(X, weights):

	X = numpy.array(X)
	W = numpy.array(weights)
	return numpy.matmul(X,W)
#-------------------------------------------------
#Function to find root mean squared error between predicted and target output
def meansquarederr(T, Tdash):
    error=0
    for index in range(0, len(T)):
        error = error + (T[index]-Tdash[index])*(T[index]-Tdash[index])
    if len(T)==0:
        return -1
    error = float(error/float(len(T)))
    return error
    
#---------------------------------------------------
# Opening the file and storing them in a matrix named 'data' and results in vector named 'output'

with open("linregdata") as f:
	for line in f:
		my_line = line.split(",")
		temp=list()

		if(my_line[0]=='F'):
			temp.extend([1.0,0.0,0.0])
		if(my_line[0]=='I'):
			temp.extend([0.0,1.0,0.0])
		if(my_line[0]=='M'):
			temp.extend([0.0,0.0,1.0])		

		for i in range(1,len(my_line)-1):
			vv = float(my_line[i]);
			temp.append(vv)
		vv = int(my_line[len(my_line)-1])
		output.append(vv)	
		data.append(temp)		
#---------------------------------------------------
#Function that returns an array of size 2 , first element is train matrix and second element is validation matrix
def sampling(data,frac):		#frac is the fraction of train data out of total data
	nu_train = int(frac*(len(data)))
	nu_vali = len(data)-nu_train

	final=list()
	final_train=list()
	final_valid=list()
	output_train=list()
	output_valid=list()


	total_index=list()
	for i in range(0,len(data)):
		total_index.append(i)
	train_index=list()
	for i in range(0,nu_train):
		ff = random.choice(total_index)
		total_index.remove(ff)
		train_index.append(ff)
	
	for i in range(0,len(data)):
		temp = standard_data[i]
		temp_output = output[i]
		if i in train_index:
			final_train.append(temp)
			output_train.append(temp_output) 
		else:
			final_valid.append(temp)
			output_valid.append(temp_output)
	final.append(final_train)
	final.append(output_train)
	final.append(final_valid)
	final.append(output_valid)
	return final


#------------------------------------------------
#Finding mean and standard deviation

mean=list()  #store the mean of the features
std = list()  #store the standard deviation of features


for j in range(0,10):
	temp=list()
	for i in range(0,len(data)):
		temp.append(data[i][j])

	feature_mean = numpy.mean(temp)
	feature_std = numpy.std(temp)

	mean.append(feature_mean)
	std.append(feature_std)
#------------------------------------------------
#standardized data

standard_data = list()

for i in range(0,len(data)):
	temp=list()
	for j in range(0,10):
		pp = data[i][j]-mean[j]
		pp = pp/std[j]
		temp.append(pp)

	standard_data.append(temp)	
#---------------------------------------------------
standard_data = [[1] + i for i in standard_data]
weights = mylinridgereg(standard_data,output,0)
predicted_output = mylinridgeregeval(standard_data,weights)

#-----------------------------------------------------

#Creating test sample (20% test)
standard_test_data = list()
standard_test_output = list()
standard_train_data = list()
standard_train_output = list()


total = len(standard_data)
test_total = int(total/5)
random_index=list()
total_index=list()
for i in range(0,total):
	total_index.append(i)

for i in range(0,test_total):
	vv=random.choice(total_index)
	total_index.remove(vv)
	random_index.append(vv)

total_index=list()
for i in range(0,total):
	total_index.append(i)


for pp in range(0,len(total_index)):
	temp = standard_data[pp]
	temp_output = output[pp]
	
	if pp in random_index:
		standard_test_data.append(temp)
		standard_test_output.append(temp_output) 
	else:
		standard_train_data.append(temp)
		standard_train_output.append(temp_output)

#------------------------------------------------------------
#Final Matrix which contains average error wrt fraction and lambda.
#rows are different fractions and coloumns are different lambdas.

final_train_mat = list()
final_test_mat = list()


#-----------------------------------------------------------

#Q6,Q7,Q8-
# list of all fractions
fractions = [0.1,0.2,0.4,0.7,0.9]
# list of all lambdas
lambds = [0.000001,0.0001,0.001,0.01,0.1,1]


for i in range(len(fractions)):
	te = list()
	te2 = list()
	for j in range(len(lambds)):
		fract = fractions[i]
		lam = lambds[j]
		train_av=0.0
		test_av=0.0
		temp_list=list()
		temp_list2=list()
		for pp in range(0,100):
			final_list = sampling(standard_train_data,fract)
			stand_training_data = final_list[0]
			stand_training_output = final_list[1]
			obtained_weights = mylinridgereg(stand_training_data,stand_training_output,lam)
			
			obtained_train_output = mylinridgeregeval(stand_training_data,obtained_weights)	
			obtained_train_error = meansquarederr(obtained_train_output,stand_training_output)
			

			obtained_test_output = mylinridgeregeval(standard_test_data,obtained_weights)	
			obtained_test_error = meansquarederr(obtained_test_output,standard_test_output)
			
			train_av=train_av+obtained_train_error
			temp_list.append(obtained_train_error)
			temp_list2.append(obtained_test_error)
			test_av = test_av + obtained_test_error
		train_av=train_av/100
		test_av = test_av/100
		te.append(train_av)
		te2.append(test_av)
		std_err = numpy.std(temp_list)
		std_test_err = numpy.std(temp_list2)
		print("For fraction = "+str(fract)+"| Lambda = "+str(lam)+"| Average Train Error = "+str(train_av)+"| Standard deviation of Train error = "+str(std_err)+"| Average Test Error = "+str(test_av)+"| Standard deviation of test error = "+str(std_test_err)+"\n")
	final_train_mat.append(te)
	final_test_mat.append(te2)	



#------------------------------------------------------------
#Q9 - 

final_list = sampling(standard_train_data,0.9)
stand_training_data = final_list[0]
stand_training_output = final_list[1]
obtained_weights = mylinridgereg(stand_training_data,stand_training_output,1)

obtained_test_output = mylinridgeregeval(standard_test_data,obtained_weights)	
file = open("tes.csv","w") 
 
list1=list()
list2=list() 
for i in range(len(obtained_test_output)):
	list1.append(standard_test_output[i])
	list2.append(obtained_test_output[i])
	
plt.scatter(list1,list2,color=['red'])
#plt.show()
#----------------------------------------------------------------



#-------------------------------------------------------------





