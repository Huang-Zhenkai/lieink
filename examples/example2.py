import numpy as np

from lieink.atoms import Twist3

# 所有公式中可出现的类型 so3, Twist3 等均为一等公民
t = Twist3(np.array([1, 2, 3]))
t.print()
# 结果如下
"""
Twist3([1., 2., 3.])
"""
# 通过toso3方法可以得到so3
s = t.toso3()
s.print()
# 结果如下
"""
so3([[ 0., -3.,  2.],
     [ 3.,  0., -1.],
     [-2.,  1.,  0.]])
"""

# 也存在SE3对应的两种伴随形式Ad，coAd
# se3对应的两种伴随形式ad，coad和旋量形式Twist，Wrench
# log/exp时可指定类型直接转化为对应实例
# 让写代码和写公式一样清晰
