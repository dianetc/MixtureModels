import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import invgamma, norm

# normal inverse chi-squared distribution
# run script via : % python3 NIX.py m k s2 v
 
class NIX:
	def __init__(self,m,k,s2,v):
		self.m = m # mean
		self.k = k # how strong we believe it
		self.s2 = s2 # variance 
		self.v = v # how strong we believe it 
	
	def inverse_X_squared(self,x,scale):
		shape = self.v/2
		scale = (self.v*scale)/2
		return invgamma.pdf(x,shape,loc=0,scale=scale)

	def plotting(self):
		m_n = 50
		s2_n = 100
		if self.m == 0:
			m0 = np.linspace(-2, 2, m_n)
		elif self.m < 0:
			m0 = np.linspace(self.m-1, (-1*self.m)+1, m_n)
		elif self.m > 0:
			m0 = np.linspace((-1*self.m)-1, self.m+1, m_n)

		s2_0 = np.linspace(0.1,self.s2, s2_n)
		ms,s2s = np.meshgrid(m0,s2_0)
		
		invchi = self.inverse_X_squared(s2s,self.s2)
		normal = norm.pdf(ms,loc=self.m,scale=np.sqrt(s2s/self.k))
		nix = np.multiply(normal,invchi)

		figure = plt.figure()

		ax = figure.add_subplot(111,projection='3d')
		ax.view_init(20,-125)
		ax.contour(ms, s2s, nix, zdir='z', offset=0, cmap=cm.Spectral)
		ax.plot_surface(ms, s2s, nix, cmap=cm.Spectral, rstride=1, cstride=1, antialiased=False)
		ax.xaxis.set_rotate_label(False)
		ax.yaxis.set_rotate_label(False)
		ax.xaxis.labelpad=10
		ax.yaxis.labelpad=10

		plt.xlabel(r'$\mu$', fontsize=10)
		plt.ylabel(r'$\sigma^2$', fontsize=10)	
		plt.show()

def main():
	m =  float(sys.argv[1])
	k =  float(sys.argv[2])
	s2 =  float(sys.argv[3])
	v = float(sys.argv[4])
	nix = NIX(m,k,s2,v)
	nix.plotting()

if __name__ == "__main__":
	main()
