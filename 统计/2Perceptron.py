import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            x1=[]                              # predicted as + sample
            x2=[]                              # predicted as - sample
            errors = 0
            for xi, target in zip(X, y):  # each sample  [2 8]  1
                print("target:",target,"predict:",self.predict(xi),"xi",xi)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                if self.predict(xi)==1: 
                    x1.append(xi)               
                else:
                    x2.append(xi)
            print("errors:",errors)
            x1=np.array(x1)
            x2=np.array(x2)
            #plt.cla()
            
            self.plot_decision_region(X,y,x1,x2,i,errors)
            #pdb.set_trace()
            '''
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 1].max() + 1
            x2_min, x2_max = X[:, 1].min() - 2, X[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
            Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)              # prediction result of each grid point  (500,550)
            plt.contourf(xx1, xx2, Z, alpha=0.4)  # yellow background, draw the seperate line  z=0
            plt.xlim(0, xx1.max())
            plt.ylim(0, xx2.max())
            
            plt.scatter(x1[:,0],x1[:,1],c='r', alpha=0.8, marker='.',label='1')  # x1  + sample point (predicted as)
            plt.scatter(x2[:,0],x2[:,1],c='g', alpha=0.8,marker='x', label='-1') # x2  - sample point
            plt.legend(loc='upper left')
            plt.title('Test')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.text(0,0,'No.%d' % (i+1),fontdict={'size':20,'color': 'red'})
            plt.text(5,0,'Errors:%d' %errors,fontdict={'size':20,'color': 'red'})
            
            plt.pause(0.1)
            plt.savefig('results/'+str(i+1)+'_result.png')'''
            
            #pdb.set_trace()
            self.errors_.append(errors)
        return self
        
    def plot_decision_region(self,X,y,x1,x2,i,errors):
        plt.cla()
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 1].max() + 1
        x2_min, x2_max = X[:, 1].min() - 2, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)              # prediction result of each grid point  (500,550)
        plt.contourf(xx1, xx2, Z, alpha=0.4)  # yellow background, draw the seperate line  z=0
        plt.xlim(0, xx1.max())
        plt.ylim(0, xx2.max())
            
        plt.scatter(x1[:,0],x1[:,1],c='r', alpha=0.8, marker='.',label='1')  # x1  + sample point (predicted as)
        plt.scatter(x2[:,0],x2[:,1],c='g', alpha=0.8,marker='x', label='-1') # x2  - sample point
        plt.legend(loc='upper left')
        plt.title('Test')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.text(0,0,'No.%d' % (i+1),fontdict={'size':20,'color': 'red'})
        plt.text(5,0,'Errors:%d' %errors,fontdict={'size':20,'color': 'red'})
            
        plt.pause(0.1)
        plt.savefig('results/'+str(i+1)+'_result.png')
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

pdb.set_trace()
x=np.array([[1,2],[1,5],[6,7],[8,9],[2,4],[7,9],[0,2],[2,8]])
y=np.array([-1,1,-1,-1,-1,1,-1,1])
p=Perceptron()
p.fit(x,y)
