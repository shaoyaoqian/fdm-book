

import sympy as sym

x, y, z, t = sym.symbols('x y z t')

u = 0
v = 0
w = 0
p = (2*x-1)*(2*y-1)*(2*z-1)*sym.exp(-t)


def laplace(u):
    return sym.diff(u, x, 2)+sym.diff(u, y, 2)


def grad(u):
    return sym.diff(u, x, 1), sym.diff(u, y, 1)

# 打印C++代码
def cpp_code(f, name='f'):
    f = sym.simplify(f) 
    f_code = sym.printing.ccode(f) 
    return 'double ' + name + ' = ' + f_code + ";"

def make_example(u):
    lambda_u = sym.lambdify([x, y], u, 'numpy')
    lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
    lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
    lambda_laplace_u = sym.lambdify([x, y], laplace(u), 'numpy')
    return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u


a = make_example(p)