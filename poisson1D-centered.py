import numpy as np
import scipy.linalg
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy as sc
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


width = 1.0
Nx = 20 
dx = width/Nx

left = 0.0
right = 0.0

uh, source, u_exact = np.zeros(Nx+2), np.zeros(Nx+2), np.zeros(Nx+2)

uh[0] = left
uh[-1] = right

# Demo 1
########################
# def cal_u_exact(x):
#     return x*(1-x)

# def cal_source(x):
#     return -2

# Demo 2
########################
def cal_u_exact(x):
    return np.sin(np.pi*x)

def cal_source(x):
    return -np.pi*np.pi*np.sin(np.pi*x)

def gauss_seidel(uh,source, Nx):
    for i in range(1, Nx+1):
        uxp = uh[i+1]
        uxn = uh[i-1]
        b = (uxp+uxn)/dx/dx
        a = -2 / (dx * dx) 
        uh[i] = (source[i] - b) / a


for i in range(0,Nx+2):
    u_exact[i] = cal_u_exact((i-0.5)*dx)
    source[i] = cal_source((i-0.5)*dx)

for i in range(1000):
    uh[0] = -uh[1]
    uh[-1] = - uh[-2]
    gauss_seidel(uh,source, Nx)
    print(np.linalg.norm(u_exact-uh, ord=2)*np.sqrt(dx))

# plotting
plt.figure()
x = np.linspace(0, width, Nx+2)
plt.plot(x, uh, label=f'uh')
plt.plot(x, u_exact, label=f'u_exact')
plt.title('1D poisson')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()