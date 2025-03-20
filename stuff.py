import methods as m
from sympy import *
x = Symbol('x')
f = 2*x**2
f_prime = f.diff(x)
print(f_prime)