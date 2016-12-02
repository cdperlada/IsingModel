
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Ising(object):
    J = 1. # interaction constant
    h = 0
    L = 100 
    k_B = 1.
    T = 2.269
    iterate = 10000 # number of iterations
    ic = 0 # to keep track of the number of iterations
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

    def on_key(self,event):
        '''To make the plot interactive.'''
        
        key = event.key
        if key == 'h':
            self.h += 1
            self.ic = 0
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
            self.title.set_text(title)
        elif key == 'g':
            self.h -= 1
            self.ic = 0
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
            self.title.set_text(title)
        elif key == 't':
            self.T += 1
            self.ic = 0
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
            self.title.set_text(title)
        elif key == 'r':
            self.T /= 2
            self.ic = 0
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
            self.title.set_text(title)
        elif key == 'i':
            self.iterate+=1
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
        elif key == 'u':
            self.iterate-=1
            title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
            
    def continue_loop(self):
        '''Create a generator.'''
        
        while self.ic < self.iterate:
            self.ic += 1
            yield self.ic 

    def update(self,continue_loop):
        '''Perform the algorithm for simulating Ising ferromagnet.'''
        
        y,x = np.random.randint(self.L),np.random.randint(self.L) # index of the randomly chosen site
        change_E = self.deltaEnergy(y,x)
        self.lattice = self.metropolis(change_E,y,x,self.lattice,self.T)
        self.im.set_array(self.lattice)
        title = 'black:-1, white:+1 \nM=%.3f \nL=%i \nT=%.3f $J/k_B$ \nh=%.3f \nnumber of iterations = %i of %i'%(self.magnetization(self.lattice),self.L,self.T,self.h,self.ic,self.iterate)
        self.title.set_text(title)

    def animate(self):
        '''Animate the Ising model.'''
        
        fig = plt.figure()        
        self.im = plt.imshow(self.lattice,cmap='gray',interpolation="nearest",animated=1)
        plt.axis('off')        
        self.title = plt.title('')
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        anim = FuncAnimation(fig,self.update,self.continue_loop,interval=10,repeat=0)
        plt.tight_layout()     
        plt.show(block=1) # use Python console or VIDLE
        #User may increase h,T, and the number of iterations as defined by the on_key function

if __name__ == "__main__":
    sim = Ising()  
    sim.animate()        