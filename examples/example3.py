import numpy as np

from lieink.atoms import SO3, Point3, Twist3
from lieink.containers import Container

rotm = SO3(np.eye(3))
rotm.rotx(0.1, inplace=True).roty(0.2, inplace=True).rotz(0.3, inplace=True)

p = Point3(np.array([1, 2, 3]))
t = Twist3(np.array([4, 5, 6]))

# %%
# atoms中的实例可以进行运算
# 如SO3*Point3,SO3*Twist3
# 通过registry_for_atoms进行注册
(rotm * p).print()
"""
Point3([0.953 , 1.9089, 3.0737])
"""
(rotm * t).print()
"""
Twist3([3.489 , 5.3884, 5.9826])
"""

# %%
# 可通过任意atoms的UPDATE_ALLOWED_OPERATORS进行注册

# 特别注意@表示伴随运算
skew_t = t.toso3()
skew_t.print()
"""
so3([[ 0., -6.,  5.],
     [ 6.,  0., -4.],
     [-5.,  4.,  0.]])
"""
(rotm @ skew_t).print()
"""
so3([[-0.    , -5.9826,  5.3884],
     [ 5.9826,  0.    , -3.489 ],
     [-5.3884,  3.489 , -0.    ]])
"""
# 打印所有允许的运算
Twist3.PRINT_ALLOWED_OPERATORS()

# %%
# Container也是通过注册表实现的create_a_container()方法的类型定向
# 通过任意Container的UPDATE_CONTAINER_TYPES进行注册
Container.PRINT_CONTAINER_TYPES()
# Container会从后向前检查实现的类

# 也就是说上述运算均可进行拓展
