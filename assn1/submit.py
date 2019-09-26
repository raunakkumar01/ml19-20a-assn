import numpy as np

import time

import random as rnd

import matplotlib.pyplot as plt

def train_test_split(X,y,ratio): # function to split given data into traing and test data
    
    n=len(y)
    
    test_size = (int) (n*ratio)
    
    return (X[0:n-test_size], y[0:n-test_size], X[n-test_size:n], y[n-test_size:n])

def Cal_hinge_squared(X,y,w): # function to calculate loss function
    
    (n,d) = X.shape
    
    l_hinge = 0
    
    for i in range(0,n):
    
        l_hinge = l_hinge + max(0,(1 - y[i]*np.dot(w,X[i])))**2
    
    return l_hinge + 0.5*np.dot(w,w)

def solver_1(X,y,C): # coordinate minimization

    (n,d) = X.shape # shape of given X
    
    y = np.reshape(y,(len(y),1)) # reshaping 
    
    data = np.append(X,y,axis=1) # joining X and y so that we can randomize data
    
    rnd.shuffle(data) # shuffling
    
    X = data[:,0:d] # new X
    
    z = np.ones((n,1), dtype=int) # np array of ones
    
    X = np.append(X,z, axis=1) # X = X + z
    
    y = data[:,d:d+1] # new y
    
    w = np.zeros( (d+1,)) #21
    
    X_train,y_train,X_test,y_test = train_test_split(X,y,0.25) # splitting data into test and training data ( hardcoded )
    
    (n,d) = X_train.shape # new shape
    
    alpha = np.zeros((n,),dtype=float) # initialize alpha
    
    q = [np.dot(x,x) for x in X_train] # q[i] = ||X[i]||**2
    
    t_end = time.time() + 15 #  run time
    
    points_x = []
    
    points_y = []
    
    tot_time = 0
    
    min_index = 0
    
    w_in_process = []
    
    while time.time() < t_end:
        
        start = time.time()
        
        for i in range(0,n): # going through data set
        
            temp = (1 - ( y[i] * ( np.dot(w, X_train[i]) - alpha[i]*y_train[i]*(q[i]))))/(q[i] + 1/(2*C)) # calculated alpha[i]
            
            #w = w + (temp - alpha[i]) * X_train[i] * y_train[i] # updating w
            
            if(temp > 0): # conditon to update
                w = w + (temp - alpha[i]) * X_train[i] * y_train[i] # updating w
                alpha[i] = temp
            
            else: # condition to update
                w = w - alpha[i] * X_train[i] * y_train[i] # updating w
                alpha[i] = 0    
                
         # updating w
        
        interval = time.time() - start #time taken by one run
        
        tot_time = tot_time + interval #time t
        #print(tot_time)
        temp_cal = Cal_hinge_squared(X,y,w) #loss value for calculated w
        
        points_y.append(temp_cal) #points on y 
        
        points_x.append(tot_time) #points on x
        
        w_in_process.append(w) # w's history
        
        if(temp_cal < points_y[min_index]):
            
            min_index = len(points_y) - 1
    
    #w_1 = points[19][2]        # best w for which loss function is minimum
    
    count = 0 # count of correct predictions
    
    #count_1 = 0 # count for correct prediction by w_1
    
    length = len(y_test) 
    
    for i in range(0,length):
    
        temp = np.dot(w,X_test[i]) # predicting y
        
        if(temp > 0): #condition
        
            temp = 1
        
        else:
        
            temp = -1
        
        if(temp == y_test[i]):
        
            count = count + 1
        
        #temp_1 = np.dot(w_1,X_test[i]) # predicting y by w_1
        
        #if(temp_1 > 0): #condition
        
         #   temp_1 = 1
        
       # else:
        
        #    temp_1 = -1
        
        #if(temp_1 == y_test[i]):
        
         #   count_1 = count_1 + 1 # count of correct prediction by w_1
    
    accuracy = (count/5000) * 100 # accuracy on test set note : test_set = 0.25 * X (its hard coded for testing purposes)
    
    #accuracy_1 = (count_1/5000)*100
    
    print(accuracy)
    
    #print(acciracy_1)
    
    b=w[d-1] # required b
    
    w = w[0:d-1] # required w
    
    return (w,b,points_x,points_y,min_index,w_in_process)




def solver_2( X, y, C ): #gradient
    (n, d) = X.shape
################################
#  Non Editable Region Ending  #
################################

    # You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

    
    y = np.reshape(y,(len(y),1)) #reshaping y  
    
    data = np.append(X,y,axis=1) # joining X and y so that we can rnadomize it
    
    rnd.shuffle(data) # shuffling
    
    X = data[:,0:d] # new shuffled X 
    
    y = data[:,d:d+1] # new Coressponding y 
    
    X = np.append(X,np.ones([n,1]), axis=1) # Appending X with [ 1 ,1 ,1 .... ,1] 
    
    (n, d) = X.shape # new shape
    
    X_train,y_train,X_test,y_test = train_test_split(X,y,0.25) # splitting data into test and training data 
    
    eta = 0.001 #step length
    
    B=100 # batch size
    
    w = np.zeros((d,),dtype=float) # w_previous 
    

################################
# Non Editable Region Starting #
################################   
    points_x = []
    
    points_y = []
    
    min_index = 0
    
    w_in_process = []
    
    tot_time = 0
################################
#  Non Editable Region Ending  #
################################
        
    t_end = time.time() + 5 # 30 second run time
        
    while(time.time() < t_end):
        # saving pervious w so that we can compare and find when to stop
        
        start = time.time()
        
        for i in range(0,n,B):
            w = w - eta*gradient(w,C,X_train[i:i+B],y_train[i:i+B]) # updating w
        
        interval = time.time() - start #time taken by one run
        
        tot_time = tot_time + interval #time t
        
        #print(tot_time)
        
        temp_cal = Cal_hinge_squared(X,y,w) #loss value for calculated w
    
        points_y.append(temp_cal) #points on y 
        
        points_x.append(tot_time) #points on x
        
        w_in_process.append(w) # w's history
        
        if(temp_cal < points_y[min_index]):
            
            min_index = len(points_y) - 1
            
    count = 0 # count of correct predictions
    
    #count_1 = 0 # count for correct prediction by w_1
    
    length = len(y_test) 
    
    for i in range(0,length):
    
        temp = np.dot(w,X_test[i]) # predicting y
        
        if(temp > 0): #condition
        
            temp = 1
        
        else:
        
            temp = -1
        
        if(temp == y_test[i]):
        
            count = count + 1
        
        #temp_1 = np.dot(w_1,X_test[i]) # predicting y by w_1
        
        #if(temp_1 > 0): #condition
        
         #   temp_1 = 1
        
       # else:
        
        #    temp_1 = -1
        
        #if(temp_1 == y_test[i]):
        
         #   count_1 = count_1 + 1 # count of correct prediction by w_1
    
    accuracy = (count/5000) * 100 # accuracy on test set note : test_set = 0.25 * X (its hard coded for testing purposes)
    
    #accuracy_1 = (count_1/5000)*100
    
    print(accuracy)
    
    
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
		# b = (b * (t-1) + b_run)/ts
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
    return (w, b, points_x,points_y,min_index,w_in_process)# This return statement will never be reached


def solver_3( X, y, C):
    (n, d) = X.shape
    
    
################################
#  Non Editable Region Ending  #
################################
    
    # You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

    
    y = np.reshape(y,(len(y),1)) #reshaping y  
    
    data = np.append(X,y,axis=1) # joining X and y so that we can rnadomize it
    
    rnd.shuffle(data) # shuffling
    
    X = data[:,0:d] # new shuffled X 
    
    y = data[:,d:d+1] # new Coressponding y 
    
    X = np.append(X,np.ones([n,1]), axis=1) # Appending X with [ 1 ,1 ,1 .... ,1] 
    
    #(n, d) = X.shape # new shape
    
    X_train,y_train,X_test,y_test = train_test_split(X,y,0.25) # splitting data into test and training data 
    
    eta = 0.000001 #step length
    (n,d) = X_train.shape #training data shape
    #B=10 # batch size
    w = np.zeros( (d,)) 
    #iterations = 0# number of iterations on training set
    len_w = len(w)
    

################################
# Non Editable Region Starting #
################################   

    points_x = []
    
    points_y = []
    
    min_index = 0
    
    w_in_process = []
    
    tot_time = 0
################################
#  Non Editable Region Ending  #
################################
        
    t_end = time.time() + 30# 30 second run time
        
    while(time.time() < t_end):
        # saving pervious w so that we can compare and find when to stop
        
        start = time.time()
################################
#  Non Editable Region Ending  #
################################
        #w = w_pre
        #while(1):
        
        #w = w_pre # wsaving pervious w so that we can compare and find when to stop
        
        #while(1):		
        for i in range(0,len_w):
            j_t=i
            w[j_t] = w[j_t] - eta*(gradient1(w,C,X_train,y_train,j_t))
            
        interval = time.time() - start #time taken by one run
        
        tot_time = tot_time + interval #time t
        
        #print(tot_time)
        
        temp_cal = Cal_hinge_squared(X,y,w) #loss value for calculated w
    
        points_y.append(temp_cal) #points on y 
        
        points_x.append(tot_time) #points on x
        
        w_in_process.append(w) # w's history
        
        if(temp_cal < points_y[min_index]):
            
            min_index = len(points_y) - 1


        #if(abs(Cal_f(w,X_train,y_train,C) - Cal_f(w_pre,X_train,y_train,C)) < 0.0001): # my converging condition
         #   break  
       # w_pre = w
    b = w[d-1] # required b 
    
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
		# b = (b * (t-1) + b_run)/ts
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
    return (w, b, points_x,points_y,min_index,w_in_process )# This return statement will never be reached


def gradient1(w,C,X,y,j_t):
    
    (n,d) = X.shape
    temp = 0
    for i in range(0,n) :
        if(y[i]*np.dot(w,X[i]) < 1):        
            temp = temp+ (1 - y[i]*np.dot(w,X[i])*(-1*y[i]*X[i][j_t]))
    	
    return w[j_t] + C*2*temp



def gradient(w,C,X,y): # function to calculate gradient
    
    (n,d) = X.shape
    
    l_hinge = np.zeros(d)
    
    batch_size = len(y)
    
    for i in range(0,batch_size):
        if(y[i]*np.dot(w,X[i]) < 1):
            l_hinge = l_hinge + (-1*y[i])*(1-y[i]*np.dot(w,X[i]))*X[i]
    
    return w + 2*C*l_hinge     


##### Code For Checking Accouracy on test set ####





Z = np.loadtxt( "data" )

y = Z[:,0]

X = Z[:,1:]

C = 1

(w,b,points_x_1,points_y_1,min_index_1,w_in_process_1) = solver_1(X,y,C)

max_points_y_1 = max(points_y_1)

min_points_y_1 = min(points_y_1)
xxx_1 = points_y_1
#points_y_1_f = [(y[0] - min_points_y_1)/(max_points_y_1-min_points_y_1) for y in points_y_1]
points_y_1_f = []
l1 = len(points_y_1)
for i in range(0,l1):
    points_y_1_f.append((points_y_1[i][0] - min_points_y_1)/(max_points_y_1-min_points_y_1))
    
max_points_x_1 = max(points_x_1)

min_points_x_1 = min(points_x_1)
points_x_1_f = []
l1 = len(points_x_1)
for i in range(0,l1):
    points_x_1_f.append((points_x_1[i] - min_points_x_1)/(max_points_x_1-min_points_x_1))
    

plt.plot(points_x_1_f,points_y_1_f,'r',label="Coordinate Maximization")

(w,b,points_x_2,points_y_2,min_index_2,w_in_process_2) = solver_2(X,y,C)

max_points_y_2 = max(points_y_2)

min_points_y_2 = min(points_y_2)
xxx_2 = points_y_2
#points_y_2_f = [(y[0] - min_points_y_2)/(max_points_y_2-min_points_y_2) for y in points_y_2]
points_y_2_f = []
l2 = len(points_y_2)
for i in range(0,l2):
    points_y_2_f.append((points_y_2[i][0] - min_points_y_2)/(max_points_y_2-min_points_y_2))

max_points_x_2 = max(points_x_2)

min_points_x_2 = min(points_x_2)
points_x_2_f = []
l2 = len(points_x_2)
for i in range(0,l2):
    points_x_2_f.append((points_x_2[i] - min_points_x_2)/(max_points_x_2-min_points_x_2))

    
plt.plot(points_x_2_f,points_y_2_f,'g',label = "Mini Batch Stochastic Descent")








(w,b,points_x_3,points_y_3,min_index_3,w_in_process_3) = solver_3(X,y,C)

max_points_y_3 = max(points_y_3)

min_points_y_3 = min(points_y_3)
xxx_3 = points_y_3
#points_y_1_f = [(y[0] - min_points_y_1)/(max_points_y_1-min_points_y_1) for y in points_y_1]
points_y_3_f = []
l3 = len(points_y_3)
for i in range(0,l3):
    points_y_3_f.append((points_y_3[i][0] - min_points_y_3)/(max_points_y_3-min_points_y_3))
    
max_points_x_3 = max(points_x_3)

min_points_x_3 = min(points_x_3)
points_x_3_f = []
l3 = len(points_x_3)
for i in range(0,l3):
    points_x_3_f.append((points_x_3[i] - min_points_x_3)/(max_points_x_3-min_points_x_3))


plt.plot(points_x_3_f,points_y_3_f,'b',label = "Coordinate Descent")

plt.legend(loc = 'upper right')
plt.xlabel("time(s)")
plt.ylabel("loss function's value")
plt.show()


    
    