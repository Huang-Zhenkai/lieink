import numpy as np

from lieink.atoms import SE3, Twist
from lieink.containers import Container

x = np.random.random((6, 1))

container = Container(Twist(x))
container.print()
l: Container[SE3] = container.exp(1)
l.print()

x = np.random.random((3, 1))
print(x)
print(x.mean(axis=0))
