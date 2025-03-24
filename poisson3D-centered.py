import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
width = 1.0
height = 1.0
depth = 1.0 
Nx = 80
Ny = 80
Nz = 80
dx = width / Nx
dy = height / Ny
dz = depth / Nz

# Boundary conditions
left, right, top, bottom, front, back = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Initialize arrays (3D)
uh = np.zeros((Nx+2, Ny+2, Nz+2))  # Solution array (including boundary points)
source = np.zeros((Nx+2, Ny+2, Nz+2))  # Source term array
u_exact = np.zeros((Nx+2, Ny+2, Nz+2))  # Exact solution array

# Apply boundary conditions
uh[0, :, :] = left
uh[-1, :, :] = right
uh[:, 0, :] = bottom
uh[:, -1, :] = top
uh[:, :, 0] = front
uh[:, :, -1] = back

# Define the exact solution and source term
def cal_u_exact(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

def cal_source(x, y, z):
    return -3 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

# Gauss-Seidel solver for 3D grid
def gauss_seidel_3d(uh, source, Nx, Ny, Nz, dx, dy, dz):
    uh[0, :, :] =  -uh[1, :, :] 
    uh[-1, :, :] =  -uh[-2, :, :] 
    uh[:, 0, :] =  -uh[:, 1, :] 
    uh[:, -1, :] = -uh[:, -2, :] 
    uh[:, :, 0] = -uh[:, :, 1]
    uh[:, :, -1] = -uh[:, :, -2]
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            for k in range(1, Nz+1):
                # Neighbors in the x, y, and z directions
                uxp = uh[i+1, j, k]  # Right neighbor
                uxn = uh[i-1, j, k]  # Left neighbor
                uyp = uh[i, j+1, k]  # Upper neighbor
                uyn = uh[i, j-1, k]  # Lower neighbor
                uzp = uh[i, j, k+1]  # Front neighbor
                uzn = uh[i, j, k-1]  # Back neighbor

                # Update the current point
                b = (uxp + uxn) / dx**2 + (uyp + uyn) / dy**2 + (uzp + uzn) / dz**2
                a = -2 / (dx**2) - 2 / (dy**2) - 2 / (dz**2)
                uh[i, j, k] = (source[i, j, k] - b) / a

# Fill exact solution and source terms
for i in range(Nx+2):
    for j in range(Ny+2):
        for k in range(Nz+2):
            x = (i-0.5) *dx
            y = (j-0.5 )*dy
            z = (k-0.5) *dz
            u_exact[i, j, k] = cal_u_exact(x, y, z)
            source[i, j, k] = cal_source(x, y, z)

# Solve using Gauss-Seidel for 40000 iterations
for iteration in range(40000):
    gauss_seidel_3d(uh, source, Nx, Ny, Nz, dx, dy, dz)
    # error = np.linalg.norm(u_exact - uh, ord='fro') * np.sqrt(dx * dy * dz)  # Frobenius norm for 3D error
    if iteration % 100 == 0:
        error = np.linalg.norm(u_exact - uh) * np.sqrt(dx * dy * dz)
        print(f"Iteration {iteration}, Error: {error}")

# 20 0.0007278627568370262
# 40 0.00018179732240585148
# 80 0.000045438817238030254
