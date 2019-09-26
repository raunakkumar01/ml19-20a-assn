import numpy as np
import random as rnd
import time as tm

def solver( X, y, C, timeout, spacing ):
    (n, d) = X.shape
    totTime = 0
    w = np.zeros( (d,)) 
    b = 0
    t=0
    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

    # You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

    #X = np.append(X,np.ones([n,1]), axis=1)
    y = np.reshape(y,(len(y),1))
    data = np.append(X,y,axis=1)
    rnd.shuffle(data)
    X = data[:,0:d+1]
    y = data[:,d:d+1]
    (n, d) = X.shape
    X_train,y_train,X_test,y_test = train_test_split(X,y,0.25) # splitting data into test and training data 
    eta = 0.001 #step length
    B=10 # batch size
    
    iterations = 0 # number of iterations on training set
    
    w_pre = np.ones((d,),dtype=float) # w_t
    

################################
# Non Editable Region Starting #
################################   
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                return (w, b, totTime)
            else:
               tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
        w = w_pre
        while(1):
        
            w_pre = w # wsaving pervious w so that we can compare and find when to stop
        
            for i in range(0,n,B):
                w = w - eta*gradient(w,C,X_train[i:i+B],y_train[i:i+B]) # updating w
        
            iterations = iterations + 1 # updating iteragtions
        
            if(abs(Cal_f(w,X,y,C) - Cal_f(w_pre,X,y,C)) < 0.1 or iterations > 10): # my converging condition
                break
    
        b=w[d-1] # required b 
    
        w = w[0:d-1] # required w

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
    return (w, b, totTime)# This return statement will never be reached



def gradient(w,C,X,y): # function to calculate gradient
    
    (n,d) = X.shape
    
    l_hinge = np.zeros(d)
    
    batch_size = len(y)
    
    for i in range(0,batch_size):
        if(y[i]*np.dot(w,X[i]) < 1):
            l_hinge = l_hinge + (-1*y[i])*(1-y[i]*np.dot(w,X[i]))*X[i]
    
    return w + 2*C*l_hinge

def train_test_split(X,y,ratio): # function to split given data into traing and test data
    
    n=len(y)
    
    test_size = (int) (n*ratio)
    
    return (X[0:n-test_size], y[0:n-test_size], X[n-test_size:n], y[n-test_size:n])


def Cal_f(w,X,y,C): # function to calculate value of f(x)
    
    (n,d) = X.shape
    
    l_hinge = 0
    
    for i in range(0,n):    
        l_hinge = l_hinge + max(1-y[i]*np.dot(w,X[i]),0)**2
    
    return 0.5*(np.linalg.norm(w)**2) + C*l_hinge            


##### Code For Checking Accouracy on test set ####
 
'''data = [] #initializing a list to import data from file

with open("data") as f: #opening and reading file line by line
    lines = f.readlines()

content =[x.strip() for x in lines] #saving each line as an element of list in conctent

data = [[float(x) for x in z.split(" ")] for z in content] # converting string to numbers

rnd.shuffle(data) # shuffling data to make it random 

data = np.array(data, dtype=float) #making np array fro  list so that we can use np functions

#(n,d) = data.shape # getting shape of data

#data = np.append(data,np.ones([n,1]), axis=1) # increasing dimension of out data to include "b" in it

X = data[:,1:] # data points

Y = data[:,0:1] # Observed Output on each data point

y = [x for x in Y[:,0]] 

y = np.array(y,dtype=int)

w,b,tottime = solver(X,y,1,3,10) # calling function to solve

X_train,y_train,X_test,y_test = train_test_split(X,y,0.25) # splitting data into test and training data 

y_predict = [] # values predicted my model

l = len(X_test)

count=0 # count of correctly predicted values

for i in range(0,l): 
    
    x = np.dot(w,X_test[i]) + b
    
    if(x < 0):
        y_predict.append(-1)
        if(y_predict[i] == y_test[i]):
            count=count+1
    else:
        y_predict.append(1)
        if(y_predict[i] == y_test[i]):
            count=count+1

accuracy = (count/5000)*100 # calulating accuracy 

print(accuracy) # accuracy'''
