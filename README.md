
readme_content = """# 🖋️ lieink

> **写李群公式像写情书一样丝滑，debug 像查作业一样无痛。**

`lieink` 是你做李群李代数研究时的那瓶**墨水**（ink）。

不是那种冷冰冰的、只会吐矩阵的工程工具，而是一个把所有数学对象都当成**一等公民**捧在手心里的贴心库。SO(3)、SE(3)、Twist、Ad、coAd……在这里它们都有自己的名字、自己的脾气，以及严格的入职体检（形状检查 + 数值检查）。

你可以先用它**快速涂鸦**验证想法，确认无误后再切换到"考试模式"高效执行。

---

## ✨ 为什么叫 lieink？

**lie**（李群李代数）+ **ink**（墨水）= **lieink**

我们希望它像一瓶好墨水：
- 拿起来就能写，不用磨墨（开箱即用）
- 写出来的字漂亮且不会晕开（严格的类型和数值检查）
- 写错了能轻松擦掉重来（Container 快速验证 → cb 高效执行的双层工作流）

---

## 🎨 核心设计理念

### 1. 公式即代码 —— 让 Python 像 LaTeX 一样优雅

在 `lieink` 里，你写的代码几乎就是黑板上的公式：

```python
from lieink.atoms import Twist3, SO3

t = Twist3([1, 2, 3])      # 黑板上的那个 t
s = t.toso3()              # 变成斜对称矩阵 so(3)
R = s.exp(intensity=1.0)   # 指数映射，嗖的一下到 SO(3)
```

log / exp 还能**指定目的地**，想去 SE3 就去 SE3，想去 Ad 就去 Ad，不用自己手动拼积木：

```python
from lieink.atoms import Twist

twist = Twist([1, 2, 3, 4, 5, 6])
G = twist.exp(1.0, expto="SE3")   # 也可以选 "Ad" 或 "coAd"
```

### 2. 人人平等 —— 所有类型都是一等公民

别的库可能把所有东西塞进一个 `numpy.ndarray` 里，让你自己记"这个 4×4 矩阵到底是 SE3 还是 Ad"。

在 `lieink`，每个对象都有自己的身份证：

| 类型 | 身份说明 |
|------|----------|
| `SO3` / `SE3` | 旋转 / 刚体变换群 |
| `so3` / `se3` | 李代数（矩阵版） |
| `Twist3` / `Twist` | 旋量（3D / 6D） |
| `Wrench` | 力旋量 |
| `Ad` / `coAd` | 伴随 / 余伴随表示 |
| `ad` / `coad` | 伴随 / 余伴随李代数 |
| `Point3` / `Point4` | 点（齐次 or not） |
| `Vector3` / `Vector4` | 向量（齐次 or not） |

它们各自有符合数学直觉的运算规则，不会搞混。

### 3. 严格的安检 —— 形状不对？数值不合法？当场拦住！

基于 `beartype` + `jaxtyping`，`lieink` 在运行时会对每个对象做**双重安检**：

```python
from lieink.atoms import SO3
import numpy as np

SO3(np.eye(4))
# ValueError: v must have shape (3,3), now (4,4).
# 安检小姐姐："先生，您的矩阵尺寸不对，请回炉重造。"

SO3(np.diag([-1, 1, 1]))
# ValueError: v is not a rotation matrix
# 安检小姐姐："行列式为 -1 也敢冒充旋转？下一个！"
```

这意味着你的 bug 会在**第一现场**被抓住，而不是默默污染下游算法，让你在深夜三点对着不对的数值怀疑人生。

### 4. 双层工作流 —— 先涂鸦，再定稿

`lieink` 提供两种画风：

- **🎨 涂鸦模式（Container）**：用 `LieContainer` 做广播、链式调用，快速验证算法逻辑。像草稿纸，涂改方便。
- **📝 定稿模式（cb 方法）**：`Twist.expcb()`、`SE3.invcb()` 等纯 NumPy 批量操作，零额外开销，适合确认后的高效执行。

```python
from lieink.atoms import Twist
from lieink.containers import LieContainer
import numpy as np

# 阶段一：草稿纸上的涂鸦
local_twists = LieContainer(Twist).extend([np.array([0,0,1,0,0,0])] * 6)
ctrls = np.random.random(6)
local_SE3 = local_twists.exp(ctrls, expto="SE3")  # 广播！丝滑！

# 阶段二：确认无误，换钢笔写正稿
local_SE3_fast = Twist.expcb(local_twists.toNDArray_3D(), ctrls, expto="SE3")
# 嗖嗖嗖，NumPy 原生广播，不带一丝感情（但很快）
```

---

## 🚀 快速开始

### 安装

```bash
git clone <你的仓库地址>
cd lieink
pip install -e .
```

> 目前源码安装即可。依赖：`numpy`, `beartype`, `jaxtyping`, `rich`, `scipy`。

### 最小示例：旋转、点、旋量，一气呵成

```python
import numpy as np
from lieink.atoms import SO3, Point3, Twist3

# 造一个旋转，链式更新，像搭积木一样爽
R = SO3(np.eye(3))
R.rotx(0.1, inplace=True).roty(0.2, inplace=True).rotz(0.3, inplace=True)

# 作用在点和旋量上
p = Point3([1, 2, 3])
t = Twist3([4, 5, 6])

(R * p).print()   # 旋转作用在点上 → 得到新点
(R * t).print()   # 旋转作用在旋量上 → 得到新旋量
(R @ t.toso3()).print()  # 伴随作用（@ 运算符）→ 得到 so3
```

### 串联机器人正运动学（POE 公式直接搬过来）

```python
from lieink.atoms import Twist
from lieink.containers import LieContainer
import numpy as np

# 假定为 6-DOF 机器人，R-R-R-P-R-R
rhos = np.random.random((6, 7))  # 初始位姿的旋量
rhos = Twist.reshapeb(rhos)      # 变成 (7, 6, 1) 的批量旋量

# 关节旋量（局部坐标系下）
local_axes = np.array([
    [0,0,1,0,0,0], 
    [0,0,1,0,0,0], 
    [0,0,1,0,0,0],
    [0,0,0,0,0,1], 
    [0,0,1,0,0,0], 
    [0,0,1,0,0,0],
    ]).T
local_twists = LieContainer(Twist).extend(local_axes)
ctrls = np.random.random(6)  # 关节变量

# 1. 涂鸦模式：快速验证逻辑
initial_pose = LieContainer(Twist).extend(rhos).exp(1.0, expto="SE3")
local_SE3 = local_twists.exp(ctrls, expto="SE3")

# 指数积：前6个初始位姿 × 局部运动，再乘最后一个初始位姿
end_pose = (initial_pose[:6] * local_SE3).prod() * initial_pose[6]
end_pose.print()

# 2. 定稿模式：确认后换 cb 方法，效率拉满
initial_fast = Twist.expcb(rhos, 1.0, expto="SE3")
local_fast = Twist.expcb(local_twists.toNDArray_3D(), ctrls, expto="SE3")

end = np.eye(4)
for i in range(6):
    end = end @ initial_fast[i] @ local_fast[i]
end = end @ initial_fast[6]

# 两者结果完全一致（可以 np.allclose 验证）
```

---

## 🏗️ 架构亮点（一些你可能想知道的幕后故事）

### 运算符注册表：一个开放的"婚介所"

`lieink` 不硬编码所有运算组合，而是通过 `ALLOWED_OPERATORS` 注册表动态管理：

```python
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", Point3, Point3)   # SO3 × Point3 → Point3
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "@", se3, se3)        # 伴随运算
```

这就像给类型们办了一个婚介所：你想介绍谁认识谁，登记一下就行，不用拆墙改地基。

### Container：一个会自我复制的广播塔

`LieContainer` 本质上是一个**类型安全的列表**，但它会魔法：

- **广播运算**：`container + other`、`container * scalar`、`container @ matrix`
- **广播方法**：`container.exp(intensities)`、`container.rotx(thetas)`
- **聚合大招**：`.prod()`（连乘）、`.sum()`（连加）、`.cumprod()`（累积连乘）
- **自动分型**：通过 `create_a_container` 自动推断返回的容器类型

你可以把它想象成一个复印机 + 广播塔：你把操作丢进去，它自动给每个元素复印一份并执行，最后把结果整整齐齐码好还给你。

### 三套 API：单例、批量、实例，各取所需

每个类型都有三种画风：

| 后缀 | 适用场景 | 示例 |
|------|----------|------|
| `c` | 单例静态计算 | `SO3.rotxc(0.5)` |
| `cb` | 批量静态计算（NumPy 广播） | `SO3.rotxcb([0.1, 0.2, 0.3])` |
| 无 | 实例方法（支持 inplace） | `R.rotx(0.5, inplace=True)` |

`cb` 方法内部全是 NumPy 原生操作，没有 Python 层 for 循环，快得飞起。

---

## 🆚 和其他库的区别

| 维度 | lieink | 典型工程库（如 sophuspy） |
|------|--------|--------------------------|
| **气质** | 研究者的草稿纸 + 钢笔 | 工程师的螺丝刀 |
| **类型系统** | 严格一等公民，运行时双重安检 | 常基于裸矩阵或轻量封装 |
| **开发体验** | 公式化、可交互验证、Container 广播 | 面向部署，API 更底层 |
| **批量计算** | 显式区分"涂鸦层"与"定稿层" | 通常只提供单一接口 |
| **可扩展性** | 运算符注册表，动态扩展 | 运算规则通常硬编码 |
| **自动微分** | 不是我们的菜（专注数值正确性） | 部分库支持 JAX/Torch AD |

如果你需要的是一个**能让你在半小时内把论文公式变成可运行代码**的工具，`lieink` 很可能是你的菜。

---

## 🎯 适用场景

- 🤖 **机器人学**：指数积（POE）运动学、雅可比分析、旋量理论验证
- 📚 **教学与论文复现**：需要严格对应数学符号、快速试错的研究流程
- 🧪 **算法验证**：先 Container 验证逻辑，再 cb 方法跑实验数据

---

## 📜 许可证

[MIT License](LICENSE)

---

> *"数学是墨水，代码是笔，lieink 是你的稿纸。"*
> 
> *—— 某个研究机器人运动学标定但无人接班的可怜研究生*

