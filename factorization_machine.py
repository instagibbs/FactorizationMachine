import theano
from theano import tensor as T
import numpy as np
from sklearn import cross_validation, datasets
from sklearn.preprocessing import normalize
import theano
import sys
import traceback
import time
import sys 

class fact_machine():
  def __init__(self, num_feats, k_size=3, ltype=2, lamb=.1):
    #num_feats = samples.shape[1]
    self.lamb = lamb
    self.ltype = ltype
    #self.num_samples=num_samples
    self.num_feats=num_feats
    self.k_size=k_size

    w0 = np.random.randn()
    self.w0 = theano.shared(value=w0, name='w0')#, borrow=True)
    
    ws = np.random.randn(num_feats)
    self.ws = theano.shared(value=ws, name='ws')#, borrow=True)
    
    vs = np.random.randn(num_feats, k_size)
    self.vs = theano.shared(value=vs, name='vs')#, borrow=True)
    
    self.input_var = T.matrix()
    self.target_var = T.vector()
    
  #Must be numpy array of float32. Keeps weights, changes other things.
  def set_data(self, x, y):
    self.shared_x = theano.shared(x)
    self.shared_y = theano.shared(y)
    
    self.givens = {
      self.input_var : self.shared_x,
      self.target_var : self.shared_y
    }
    self.set_updates()
    self.set_train()
    self.set_output()
    
  #Defines the inference-level objective section.
  def factorization_objective(self, samples):
    yhat = self.w0+T.dot(samples,self.ws)
    for i in range(self.num_feats-1):
      for j in range(i+1,self.num_feats):
        yhat += T.dot(self.vs[i], self.vs[j])*samples[:,i]*samples[:,j]
    return yhat
    
  #The exponential penalty objective outlined in: http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf
  def exp_objective(self):

    total_objective = T.log(1+T.exp(-self.target_var*self.factorization_objective(self.input_var)))
    if self.ltype == 2:
      total_objective += (self.lamb/2)*T.sum(T.sqr(self.ws)) 
      total_objective += (self.lamb/2)*T.sum(T.sqr(self.vs))
    elif self.ltype == 1:
      total_objective += self.lamb*T.sum(T.abs_(self.ws))
      total_objective += self.lamb*T.sum(T.abs_(self.vs))
    else:
      raise Exception('Wrong regularization type, must be 1 or 2: ' + str(ltype))
    return T.mean(total_objective)
    
  #SGD formmulation
  def gen_updates_sgd(self, loss, learning_rate=.1):
    all_parameters = [self.w0, self.ws, self.vs]
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
      updates.append((param_i, param_i - learning_rate * grad_i))
    return updates
    
  def set_updates(self):
    updates = self.gen_updates_sgd(self.error())
    self.updates = updates
  
  def error(self):
    return self.exp_objective()
  
  def predict(self):
    output = self.factorization_objective(self.input_var)
    return output
    
  def set_train(self):
    self.train = theano.function([], self.error(), givens=self.givens, updates=self.updates) 
    
  def set_output(self):
    self.output = theano.function([], self.predict(), givens=self.givens, on_unused_input='ignore')
'''
iris = datasets.load_iris() #junk dataset for now
samples = iris.data
samples = normalize(samples)
labels = iris.target
labels[labels > 0] = 1
labels[labels == 0] = -1

train_x, test_x, train_y, test_y = cross_validation.train_test_split(samples, labels, test_size=0.10)
train_x_f=train_x.astype(np.float32)
train_y_f=train_y.astype(np.float32)
test_x_f=test_x.astype(np.float32)
test_y_f=test_y.astype(np.float32)

#minibatch_size = 100
mb_size = train_x.shape[0]
print mb_size
num_batches = train_x.shape[0]/mb_size
rand_vec = np.arange(mb_size)
#np.random.shuffle(rand_vec)

f_m = fact_machine(train_x.shape[1], mb_size, k_size=0)
f_m.set_data(train_x_f, train_y_f)


converged = False
last_perf = 10000000000
while not converged:
  perf_vec =  f_m.train()
  print perf_vec
  if last_perf <= perf_vec + .000001:
    converged = True
  last_perf = perf_vec

b = 3  
print "Training Accuracy:", np.count_nonzero(np.sign(f_m.output()-f_m.w0.eval()) == train_y), len(train_y)
f_m.set_data(test_x_f, test_y_f)
print "Testing Accuracy:", np.count_nonzero(np.sign(f_m.output()-f_m.w0.eval()) == test_y), len(test_y)
'''