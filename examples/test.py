import numpy as np

from lieink.atoms import Twist
from lieink.utils import print4

x = np.random.random((6, 1))

ljacobian = Twist.left_jacobiancb(x)
rjacobian = Twist.right_jacobiancb(x)

print4(ljacobian @ np.linalg.inv(rjacobian))
print4(Twist.expcb(x, 1, expto="Ad"))

