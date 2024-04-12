import random

def square(x):
    '''Return the square of x.'''

    return x*x

def square_derivative(x):
    '''Return the gradient of the square function at x.'''

    return 2*x


x0 = 4
_lambda = 0.05
print("Initial x = {}".format(x0))
print("--------------------------")
for epoch in range(10):
    for _ in range(10):
        x0 -= square_derivative(x0) * _lambda
    print(f"epoch:{epoch} x = {x0}")