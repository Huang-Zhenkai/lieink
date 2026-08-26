import numpy as np

from lieink.atoms import SO3

rotm1 = np.eye(3)
rotm2 = np.eye(4)
rotm3 = np.eye(3)
rotm3[0, 0] = -1

# %%
# 1、atoms的实例化
# 直接在初始化时给定
rotm1_SO3 = SO3(rotm1)
# print()方法使用rich库进行精美打印
rotm1_SO3.print()
# 结果如下
"""
SO3([[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]])
"""
# %%
# 可以使用assign方法进行指定，一个实例只能指定一次
rotm1_SO3 = SO3()
rotm1_SO3.assign(rotm1)
rotm1_SO3.print()
# 结果如下
"""
SO3([[1., 0., 0., 0.],
     [0., 1., 0., 0.],
     [0., 0., 1., 0.],
     [0., 0., 0., 1.]])
"""
# %%
# 再次assign(rotm1)会报错
rotm1_SO3.assign(rotm1)
# ValueError: v of SO3 has been assigned already

# %%
# 2、atoms使用beartype进行严格的形状检查
rotm2_SO3 = SO3(rotm2)
# ValueError: v must have shape (3,3), now (4,4).

# %%
# 3、atoms在内部进行了严格的数值检查
rotm3_SO3 = SO3(rotm3)
# ValueError: v is not a rotation matrix

# %%
# 4、atoms使用v属性调用内部储存的数组
print(rotm1_SO3.v)
