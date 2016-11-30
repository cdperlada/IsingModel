
import numpy as np
import matplotlib.pyplot as plt

class Ising(object):
    J = 1.0 #interaction constant
    h = 0
    L = 101 # lattice linear dimension
    k_B = 1.0
    T = 2.269
    lattice = np.random.choice([1,-1],size=[L,L])
    
    def magnetization(self,lattice):
        '''It determines the magnetization'''
        
        return np.sum(lattice)/self.L

    def metropolis(self,change_E,y,x,lattice,T):
        '''It decides to flip the state using the Metropolis formula'''
        
        r = np.random.random()
        if r < np.exp(change_E/(self.k_B*T)):
            lattice[y,x] = -lattice[y,x]
        #note that if change_E>=0, r is always < np.exp(change_E/(k_B*T))
        return lattice

    def deltaEnergy(self,y,x):
        '''It calculates the change in energy after flipping a random state.'''
        # periodic lattice
        per = np.empty([self.L+2,self.L+2],dtype=int) 
        per[1:self.L+1,1:self.L+1] = self.lattice
        per[0,1:self.L+1] = self.lattice[self.L-1]
        per[self.L+1,1:self.L+1] = self.lattice[0]
        per[1:self.L+1,0] = self.lattice[:,self.L-1]
        per[1:self.L+1,self.L+1] = self.lattice[:,0]
        
        X = x+1
        Y = y+1
        S_j = per[Y-1,X]+per[Y+1,X]+per[Y,X+1]+per[Y,X-1]
        h_i = self.J*S_j + self.h
        return -2*self.lattice[y,x]*h_i

    def iteration(self):
        # number of iterations
        iterate = 10000 
        for _ in xrange(iterate):    
            # index of the randomly chosen site
            y,x = np.random.randint(self.L),np.random.randint(self.L) 
            change_E = self.deltaEnergy(y,x)
            self.lattice = self.metropolis(change_E,y,x,self.lattice,self.T)
            
        plt.gray()
        plt.matshow(self.lattice)
#        plt.axis('off')
#        pamagat = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f'%(self.M(self.lattice),self.L,self.T,self.h)
        plt.title('Ising Model')

if __name__ == "__main__":
    sim = Ising()  
    sim.iteration()        