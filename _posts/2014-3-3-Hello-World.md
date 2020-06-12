---
layout: default
title: test post
---

Next you can update your site name, avatar and other options using the $$\int_{-\infty} 2 + 2 = 4 $$



The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.




```python
import numpy as np
import math
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt



def x_mean(x_prev,t):
    return x_prev/2 + 25*(x_prev)/(1+x_prev**2) + 8*math.cos(1.2*t)

def y_mean(x_new):
    return x_new**2 / 20

def draw_sample(t,x_prev,system_variance,state_variance):
    x_new = np.random.normal(x_mean(x_prev,t),math.sqrt(state_variance),1)
    y_new = np.random.normal(y_mean(x_new), math.sqrt(system_variance),1)
    return np.array([x_new, y_new]).reshape(2,)

# Initialize Weights 
def initialize_weights(N,y_initial,system_variance):
    x_prior = ss.norm.rvs(x_mean(y_initial,0),np.sqrt(10),N)
    weights = ss.norm.pdf(y_initial,y_mean(x_prior),np.sqrt(system_variance))
    return(weights,x_prior)

# Bootstrap Filter with resampling each timestep
def SMC(y,initial_x, initial_weights,N,T,system_variance,state_variance):
    weights = {0: initial_weights}
    x = {0: initial_x}
    for t in range(1,T):
        
        x_prev = initial_x
        w_prev = initial_weights
        x_prev_resample = np.random.choice(x_prev,N,p=w_prev)
    
        x_new = ss.norm.rvs(x_mean(x_prev_resample,t),np.sqrt(state_variance),N)
        w_new = ss.norm.pdf(y[t],y_mean(x_new),np.sqrt(system_variance))
        
        initial_x = x_new
        initial_weights = w_new / w_new.sum()
        
        weights[t] = initial_weights
        x[t] = initial_x
        
    return weights,x


def particle_smoother_pass(weights,x,state_variance):
    
    T = len(x) - 1
    x_smooth = np.empty([T+1])
    x_smooth[T] = x[T][np.random.choice(range(weights[T].size),1,replace=True,p=weights[T])]
    
    for t in range(T-1,-1,-1):
        current_weights = weights[t]
        current_x = x[t]
        rho_unormalized = np.multiply(current_weights,ss.norm.pdf(x_smooth[t+1],x_mean(current_x,t+1),np.sqrt(state_variance)))
        x_smooth[t] = current_x[np.random.choice(range(rho_unormalized.size),1,replace=True,p=rho_unormalized / rho_unormalized.sum())]
    return x_smooth
        

def particle_smoother(N,weights,x,state_variance):
    trajections = pd.DataFrame(columns = range(len(x)))
    for n in range(N):
        trajections.loc[n] = particle_smoother_pass(weights,x,state_variance)
    return trajections




system_var = 1
state_var = 10

df = pd.DataFrame(columns=('State', 'System'))

df.loc[0] = draw_sample(0,np.random.normal(0,10,1),system_var,state_var)

for i in range(1,100):
    df.loc[i] = draw_sample(i,df.at[i-1,'State'],10,1)
    
y= df['System'].values

#df['State'].plot()
```

{% include lib/mathjax.html %}
