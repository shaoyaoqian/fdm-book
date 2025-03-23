import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
width = 1.0
height = 1.0
Nx = 40 
Ny = 40
dx = width/Nx
dy = width/Ny

# Boundary conditions
left, right, top, bottom = 0.0, 0.0, 0.0, 0.0

# Initialize arrays
uh = np.zeros((Nx+1, Ny+1))  # Solution array (including boundary points)
source = np.zeros((Nx+1, Ny+1))  # Source term array
u_exact = np.zeros((Nx+1, Ny+1))  # Exact solution array

# Apply boundary conditions
uh[0, :] = left
uh[-1, :] = right
uh[:, 0] = bottom
uh[:, -1] = top

# Define the exact solution and source term
def cal_u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def cal_source(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Gauss-Seidel solver for 2D grid
def gauss_seidel_2d(uh, source, Nx, Ny, dx, dy):
    for i in range(1, Nx):
        for j in range(1, Ny):
            # Neighbors in the x and y direction
            uxp = uh[i+1, j]  # Right neighbor
            uxn = uh[i-1, j]  # Left neighbor
            uyp = uh[i, j+1]  # Upper neighbor
            uyn = uh[i, j-1]  # Lower neighbor
            
            # Update the current point
            b = (uxp + uxn) / dx**2 + (uyp + uyn) / dy**2
            a = -2 / (dx**2) - 2 / (dy**2)
            uh[i, j] = (source[i, j] - b) / a

# Fill exact solution and source terms
for i in range(Nx+1):
    for j in range(Ny+1):
        x = i / Nx
        y = j / Ny
        u_exact[i, j] = cal_u_exact(x, y)
        source[i, j] = cal_source(x, y)

# Solve using Gauss-Seidel for 1000 iterations
for iteration in range(40000):
    gauss_seidel_2d(uh, source, Nx, Ny, dx, dy)
    error = np.linalg.norm(u_exact - uh, ord='fro')*np.sqrt(dx*dy)  # Frobenius norm for 2D error
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Error: {error}")
