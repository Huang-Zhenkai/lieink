import numpy as np

from lieink import print4
from lieink.atoms import Twist
from lieink.containers import LieContainer

# 基于局部指数积的串联机器人正运动学实现
# 假定为6-DOF机器人，R-R-R-P-R-R

rhos_data = np.random.random((6, 7))  # 初始位姿的李代数形式
joints_type = ["R", "R", "R", "P", "R", "R"]  # 每个关节的类型

rhos_data = Twist.reshapeb(rhos_data)  # 转换为Twist序列: (7, 6, 1)
local_twists_data = np.zeros((6, 6))

for i in range(local_twists_data.shape[1]):
    if joints_type[i] == "R":
        local_twists_data[2, i] = 1
    else:
        local_twists_data[5, i] = 1
print(local_twists_data)

"""
[[0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [1. 1. 1. 0. 1. 1.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]]
"""

# 1.使用LieContainer类来表示Twist序列,

# 指定容器容纳的类型
initial_pose_twists = LieContainer(Twist)
# 所有的Container本质是一列表，extend函数可以很方便地添加数据与转化
initial_pose_twists.extend(rhos_data)
# print()方法使用rich库进行精美打印
initial_pose_twists.print()
# 结果如下
"""
LieContainer[Twist]
[
│ Twist([0.5893, 0.7612, 0.4031, 0.9197, 0.6205, 0.5594]),
│ Twist([0.8172, 0.0065, 0.3329, 0.8257, 0.5665, 0.5424]),
│ Twist([0.946 , 0.5653, 0.2799, 0.8657, 0.6376, 0.7172]),
│ Twist([0.0299, 0.19  , 0.4328, 0.8367, 0.864 , 0.1581]),
│ Twist([0.1513, 0.1481, 0.2973, 0.4646, 0.4491, 0.1409]),
│ Twist([0.8539, 0.8356, 0.0993, 0.4165, 0.3136, 0.6989]),
│ Twist([0.6548, 0.261 , 0.5793, 0.7118, 0.8382, 0.5309])
]
"""

# 与原生list不同，容器支持链式调用
local_twists = LieContainer(Twist).extend(local_twists_data)
local_twists.print()
# 结果如下
"""
LieContainer[Twist]
[
│ Twist([0., 0., 1., 0., 0., 0.]),
│ Twist([0., 0., 1., 0., 0., 0.]),
│ Twist([0., 0., 1., 0., 0., 0.]),
│ Twist([0., 0., 0., 0., 0., 1.]),
│ Twist([0., 0., 1., 0., 0., 0.]),
│ Twist([0., 0., 1., 0., 0., 0.])
]
"""

# Container通过__getattr__()方法实现列表的广播运算，内部使用for循环，效率低
# 生成各关节的运动量
ctrls = np.random.random(6)
# Twist可通过exp()方法映射至李群SE3, Ad, coAd
# 上述均为atoms中的一等公民
local_SE3 = local_twists.exp(ctrls, expto="SE3")
# 输入参数也可被广播
initial_pose_SE3 = initial_pose_twists.exp(1, expto="SE3")
# 是的，运算也可以进行广播
end_pose = (initial_pose_SE3[:6] * local_SE3).prod() * initial_pose_SE3[6]
end_pose.print()
# 结果如下
"""
SE3([[ 0.2727, -0.542 ,  0.7949,  2.2932],
     [ 0.6185, -0.5341, -0.5764,  3.2958],
     [ 0.737 ,  0.6488,  0.1896,  4.8685],
     [ 0.    ,  0.    ,  0.    ,  1.    ]])
"""

# 2、确认无误后使用atoms中的XXcb方法，其中使用numpy广播实现，效率更高
initial_pose_SE3_ = Twist.expcb(rhos_data, 1, expto="SE3")
local_SE3_ = Twist.expcb(local_twists_data, ctrls, expto="SE3")
# 生成三维数组，(n, 4, 4)
end_pose_ = np.eye(4)
for i in range(6):
    end_pose_ = end_pose_ @ initial_pose_SE3_[i] @ local_SE3_[i]
end_pose_ = end_pose_ @ initial_pose_SE3_[6]
print4(end_pose_)
# 结果如下
"""
[[-0.4109  0.5222  0.7473  3.5911]
 [-0.2448 -0.8528  0.4613  0.0711]
 [ 0.8782  0.0066  0.4783  5.6932]
 [ 0.      0.      0.      1.    ]]
"""
# 使用v属性可把atoms中储存的数组导出
print(np.allclose(end_pose.v, end_pose_))
# True

# 3、实际上LieContainer也可以转化为numpy数组
initial_pose_SE3 = initial_pose_SE3.toNDArray_3D()
print(np.allclose(initial_pose_SE3_, initial_pose_SE3))
# 也存在.toNDArray_2D()方法，可进行行'r','c','d'的转换
# 或者.toLie()方法，将.toNDArray_2D()的结果转化为Lie，支持与atoms中的任意类型进行运算，结果也为Lie

