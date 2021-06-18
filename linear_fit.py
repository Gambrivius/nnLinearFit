import matplotlib.pyplot as plt
import numpy as np
import time
# line fit experiment
# use a single neuron to fit a line

# line definition format y = mx+b
m = 2
b = 15

# plot line for reference
x0 = np.linspace(-15, 15, 1000)
y0 = m*x0 + b
print ("Defined line in y=%sx+%s"% (m, b))
plt.plot(x0, y0);
plt.show()
time.sleep(2.0)

# In[201]:


# generate a random scatter plot based around the defined line.
x1 = np.linspace(-15, 15, 100)
y1 = x1*m +b+ np.random.randn(100)*10
true_y = x1*m +b
plt.scatter(x1, y1)
plt.plot(x1, true_y)
plt.show()

time.sleep(2.0)
# In[202]:



# our one neuron network
weight = np.random.rand()
bias = np.random.rand()

print ("Initial values of single neuron")
print ("Input weight:  ", weight)
print ("Bias: ", bias)


# In[203]:



def predict_y(x):
    # predict y for input x
    # x is the input to the neuron
    return x*weight + bias


def mse ():
    # calculated the mse for all samples
    p_y = predict_y(x1)
    error = p_y - true_y
    
    # return the mean square error
    return np.mean(error**2)


# In[204]:


def gradient():
    # returns a tuple (dxw,dxb)
    # dxw is the mean derivative of the weights
    # dxb is the mean derivative of the bias
    
    # return the mean gradient of weight
    p_y = predict_y(x1)
    
    # derivative of the cost function with respect to the activation function
    dxc = 2*(p_y - true_y)
    
    # derivative of the activation with respect to the weight
    dxa = x1
    
    # return the dxw and dxb
    
    dxw = np.mean(dxc*dxa)
    dxb = np.mean (dxc)
    return (dxw, dxb)


# In[205]:

def fit():
    global weight, bias
    dxw, dxb = gradient()
    weight = weight - dxw * lr
    bias = bias - dxb *lr

def drawplot():
    py = predict_y(x1)
    plt.scatter(x1, y1)
    plt.plot(x1, py)
    plt.plot(x0, y0)
    plt.show()
    
lr = 0.001 # learning rate

dxw, dxb = gradient()




for epoch in range (20):
    for n in range (100):
        fit()
        if n <= 15 and epoch == 0:
            drawplot()
            time.sleep (0.2) # sleep between fits on first epoch because slope 
                             # converges quickly. for animation purpose only
    drawplot()
    time.sleep(0.2)
    
    print ("Epoch ", epoch)
    print ("Error: ", mse())
    
    
    


# In[206]:



py = predict_y(x1)
plt.scatter(x1, y1)
plt.plot(x1, py)
plt.plot(x0, y0)
plt.show()


# In[174]:


print ("Defined line in y=%sx+%s"% (m, b))

print ("Learned line in y=%sx+%s"% (weight, bias))


