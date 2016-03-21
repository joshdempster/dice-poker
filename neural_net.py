# ----------------------------------------------------------------------------
# Copyright (c) 2016 Joshua Dempster
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------


import numpy as np
from time import clock
from math import exp
import random

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

class NeuralNet:
    '''A simple, feed-forward network. Supports backpropogation'''
    def __init__(self, layers):
        '''`layers`: iterable of int, gives the network shape'''
        self.matrices = []
        self.nodes = []
        self.nlayers = len(layers)
        
        for i, layer in enumerate(layers):
            self.nodes.append(np.zeros( layer ))
            self.nodes[-1][0] = 1.0
        for i in range(1, self.nlayers):
            self.matrices.append( 10*(np.random.random( (layers[i-1], layers[i]))-.5 )/
                                   (layers[i-1]) )
            if i < self.nlayers - 1:
                self.matrices[-1][:, 0] = 0.0

        self.Gradient = [np.copy(matrix) for matrix in self.matrices]
        self.Delta = [np.copy(node) for node in self.nodes][1:] #note: 1 shorter than nodes!
        self.j = 0.0

    def randomize(self, alpha):
        ''' randomize connection weights. Alpha sets the variance'''
        for matrix in self.matrices:
            matrix = matrix + alpha*np.random.random(matrix.shape)

    def lower_bias(self, scale):
        '''use after a large randomization to avoid biasing outputs'''
        for matrix in self.matrices:
            matrix[0,:] *= scale

    def predict(self, X):
        '''`X`: single-layer np.array (input layer)'''
        self.nodes[0][1:] = X
        for i in range(1, self.nlayers):
            self.nodes[i] = 1.0/(1.0 +np.exp(-self.nodes[i-1].dot(self.matrices[i-1])))
            if i < self.nlayers - 1:
                self.nodes[i][0] = 1.0
        return self.nodes[-1]

    def set_synapses(self, n, values):
        self.matrices[n] = values

    def back_propogate(self, data, indices=None, regularization=0):
        '''sets Delta attribute to the derivative of the cost function.

        Parameters:
            `data`: list of (np.array, np.array), where first entry is the input, second is desired ouput
            `indices`: iterable of int. Which elements of data to use for backpropogation
            `regularization`: cost of connection weights
            '''

        for grad in self.Gradient:
            grad[:] = 0.0
        for delta in self.Delta:
            delta[:] = 0.0

        self.j = 0.0
        if not indices:
            indices = range(len(data))
        m = len(indices)
        for i in indices:
            x, y = data[i]
            p = self.predict(x)
            self.j += np.sum(np.square(self.nodes[-1] - y))
            self.Delta[-1] = (self.nodes[-1] - y)*self.nodes[-1]*(1-self.nodes[-1])
            for i in range(2, len(self.nodes)):
                self.Delta[-i] = (self.matrices[-i+1].dot(self.Delta[-i+1]))*self.nodes[-i]*(1-
                                                                                              self.nodes[-i])
            for i, grad in enumerate(self.Gradient):
                grad += np.outer(self.nodes[i], self.Delta[i]/m)
                if regularization:
                    #regularizes bias inputs as well - should not make a big difference for
                    #normalized data
                    grad += regularization * self.matrices[i]
                
        self.j *= .5/m

    def train(self, data, alpha, max_iterations=1,
              indices=None, regularization=0, verbose=False):
        '''Use gradient descent on the specified data set.
            Parameters:
                `data`: as in back_propogate
                `alpha`: learning rate
                `max_iterations`: how many iterations to train for
                `indices`: as in back_propogate
                `regularization`: as in back_propogate
                '''
        
        for i in range(max_iterations):
            cost = self.j
            self.back_propogate(data, indices, regularization)
            for grad, matrix in zip(self.Gradient, self.matrices):
                matrix -= alpha*grad
            if not i%10 and verbose:
                print 'iteration: %i cost: %f' %(i, self.j)
            change = abs(self.j-cost)/self.j
            if i == max_iterations -1:
                if verbose: print 'max iterations reached. Last change: %f  iterations: %i  cost: %f ' %(
                    abs(self.j-cost)/self.j, i, self.j)

    def cost(self, data):
        '''cost without regularization'''
        J = 0
        m = len(data)
        for X, Y in data:
            p = self.predict(X)
            J += np.sum(np.square(self.nodes[-1] - Y))
        return .5*J/m

    def check_grad(self, data, epsilon=.0001):
        ''' numerically compute gradient by varying the matrix element at val_to_check by epsilon.
        Parameters:
            `data`: list of (np.array, np.array)
            `epsilon`: small float (10^-4 or so)
            '''

        gradient = []

        for matrix in self.matrices:
            gradient.append(np.zeros(matrix.shape, np.float64))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
        
                    original = matrix[i, j]
                    matrix[i, j] += epsilon
                    Jplus = self.cost(data)
            
                    matrix[i, j] -= 2*epsilon
                    Jminus = self.cost(data)
            
                    matrix[i, j] = original
                    gradient[-1][i,j] = (Jplus-Jminus)/(2*epsilon)

        return gradient

    def save(self, name):
        '''simple plain-text save format'''
        with open(name, 'w+') as f:
            for node in self.nodes:
                f.write('%i '%len(node))
            f.write('\n')
            for matrix in self.matrices:
                for line in matrix[:]:
                    for entry in line[:]:
                        f.write('%f '%entry)
                    f.write('\n')


def load_net(name):
    with open(name) as f:
        shape = [int(s) for s in f.readline().split(' ')[0:-1]]
        net = NeuralNet(shape)
        for matrix in net.matrices:
            for row in matrix:
                row[:] = [float(s) for s in f.readline().split(' ')[0:-1]]
        return net
        

if __name__ == '__main__':
    #simple test with AND and NAND functions
    #note: python is very inefficient for such a small network, but for larger networks numpy
    #   is the limiting factor, and it's fast
    net = NeuralNet((3, 2))
    and_data = []
    for i in range(1000):
        x = np.random.random(2)
        if np.all(x > .5):
            y = 1.0
        else:
            y = 0.0
        if y==1 or (y==0 and random.random()<1.0/3):
            and_data.append( (x, np.array([y, 1-y])) )
    net.train(and_data, alpha=5.0, regularization=0.0, max_iterations=2000, verbose=True)
    net.save('test_net.dat')
    net = load_net('test_net.dat')
    for i in range(10):
        x = np.random.random(2)
        print 'input: %1.2f %1.2f, output: %1.2f %1.2f' %tuple(x.tolist() +
                                                          net.predict(x).tolist())
        
            
    
