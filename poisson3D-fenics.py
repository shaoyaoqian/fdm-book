from dolfin import *
import numpy as np

N = (4,4,4)
N = (8,8,8)
N = (16,16,16)
N = (32,32,32)
N = (64,64,64)
# N = (128,128,128)
L = (2.0,2.0,2.0)
# x0 = - L[0]/N[0]/2.0
# y0 = - L[1]/N[1]/2.0
# z0 = - L[2]/N[2]/2.0
# x1 = L[0]/N[0]/2.0 + L[0]
# y1 = L[1]/N[1]/2.0 + L[1]
# z1 = L[2]/N[2]/2.0 + L[2]
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = L[0]
y1 = L[1]
z1 = L[2]
p0 = Point(x0,y0,z0)
p1 = Point(x1,y1,z1)
mesh = BoxMesh(p0,p1,N[0],N[1],N[2])
V = FunctionSpace(mesh, "Lagrange", 1)
bc = DirichletBC(V, Constant(0.0), "on_boundary")
u = TrialFunction(V)
v = TestFunction(V)

# f = Expression("3*pi*pi/2.0/2.0*sin(x[0]*pi/(x1-x0)) * sin(x[1]*pi/(y1-y0)) * sin(x[2]*pi/(z1-z0))", degree=4, x0=x0,x1=x1,y0=y0,y1=y1,z0=z0,z1=z1)
f = Expression("2*(x[1]*(2.-x[1])*x[2]*(2.-x[2])+x[0]*(2.-x[0])*x[2]*(2.-x[2])+x[0]*(2.-x[0])*x[1]*(2.-x[1]))", degree=4, x0=x0,x1=x1,y0=y0,y1=y1,z0=z0,z1=z1)


ff = Function(V)
ff.interpolate(f)

a = inner(grad(u), grad(v))*dx
LL = f*v*dx
A = assemble(a)
b = assemble(LL)

u = Function(V)
# u.interpolate(f)
# solve(a == LL, u, bc, "cg", "hypre_amg")
bc.apply(A)
bc.apply(b)
solve(A, u.vector(), b,  "cg", "hypre_amg")
for i in range(N[0]):
    p = (x0 + (i + 0.5)*L[0]/N[0], y0 + L[1]/2+L[1]/N[1]/2, z0 + L[2]/2+L[2]/N[2]/2)
    # print(u(p))
    print(ff(p))
# print(p)

# u_exact = Expression("sin(x[0]*pi/(x1-x0)) * sin(x[1]*pi/(y1-y0)) * sin(x[2]*pi/(z1-z0))", degree=4, x0=x0,x1=x1,y0=y0,y1=y1,z0=z0,z1=z1,pi=np.pi)
u_exact = Expression("x[0]*(2.-x[0])*x[1]*(2.-x[1])*x[2]*(2.-x[2])", degree=4, x0=x0,x1=x1,y0=y0,y1=y1,z0=z0,z1=z1,pi=np.pi)


print(assemble((u_exact-u)*(u_exact-u)*dx))
file = File("poisson.pvd")
file << u