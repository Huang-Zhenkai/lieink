# 🖋️️ lieink

> **写李群公式像写情书一样丝滑，搭机器人机构像拼积木一样直觉，debug 像查作业一样无痛。**

`lieink` 是你做李群李代数与机器人运动学研究时的那瓶**墨水**（ink）。

不是那种冷冰冰的、只会吐矩阵的工程工具，而是一个把所有研究对象——从 SO(3)、SE(3)、Twist、Ad 这些**数学原子**，到串联臂、并联机构、Stewart 平台这些**机构整机**——都当成一等公民捧在手心里的贴心库。每个对象都有自己的名字、自己的脾气，以及严格的入职体检（形状检查 + 数值检查）。

你可以先用它**快速涂鸦**验证想法，确认无误后再切换到"考试模式"高效执行；想折腾新构型，继承基类、填几行血肉就能跑。

---

## ✨ 为什么叫 lieink？

**lie**（李群李代数）+ **ink**（墨水）= **lieink**

我们希望它像一瓶好墨水：

- 拿起来就能写，不用磨墨（开箱即用：POE 正解、Stewart 封闭逆解都是一行调用）
- 写出来的字漂亮且不会晕开（严格的类型和数值检查）
- 写错了能轻松擦掉重来（Container 快速验证 → cb 高效执行的双层工作流）
- 墨色还能自己调（基类骨架 + 运算符注册表，扩展新类型、新构型，登记一下就行）

---

## 🎨 核心设计理念

### 1. 公式即代码 —— 让 Python 像 LaTeX 一样优雅

在 `lieink` 里，你写的代码几乎就是黑板上的公式：

```python
import numpy as np
from lieink.atoms import Twist3, SO3

# 注意：所有的 atoms 初始化都需要使用 numpy 数组，不允许使用列表
t = Twist3(np.array([1, 2, 3]))      # 黑板上的那个 t
s = t.toso3()                        # 变成斜对称矩阵 so(3)
R = s.exp(intensity=1.0)             # 指数映射，嗖的一下到 SO(3)
```

log / exp 还能**指定目的地**，想去 SE(3) 就去 SE(3)，想去 Ad 就去 Ad，不用自己手动拼积木：

```python
import numpy as np
from lieink.atoms import Twist

# 同样，必须传入 numpy 数组，不允许使用列表
twist = Twist(np.array([1, 2, 3, 4, 5, 6]))
G = twist.exp(1.0, expto="SE3")   # 也可以选 "Ad" 或 "coAd"
```

### 2. 人人平等 —— 原子与机构，皆为一等公民

别的库可能把所有东西塞进一个 `numpy.ndarray` 里，让你自己记"这个矩阵到底是 SE3 还是 Ad"。

在 `lieink`，每个数学对象都有自己的身份证：

| 类型 | 身份说明 |
| --- | --- |
| `SO3` / `SE3` | 旋转 / 刚体变换群 |
| `so3` / `se3` | 李代数（矩阵形式） |
| `Twist3` / `Twist` | 旋量（3D / 6D） |
| `Wrench` | 力旋量 |
| `Ad` / `coAd` | 伴随 / 余伴随表示 |
| `ad` / `coad` | 伴随 / 余伴随李代数 |
| `Point3` / `Point4` | 点（齐次 or not） |
| `Vector3` / `Vector4` | 向量（齐次 or not） |

**机构对象**同样如此——它们不是原子们的"示例代码"，而是有完整身份的一等公民：

| 类型 | 身份说明 |
| --- | --- |
| `SerialMechanism` | POE 串联机构：正运动学 + 空间速度雅可比，单条 / 批量（`b`）双版本 |
| `Limb` | 支链：带驱动语义的串联机构，可一键求约束旋量 |
| `ParallelMechanism` | 并联机构：Newton 迭代闭环协调求解，批量求解带逐行收敛掩码 |
| `StewartPlate` | 6-UPS Stewart 平台：给铰链几何即自动生成支链，含封闭逆解 |

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

构造机构时同样一丝不苟：

```python
SerialMechanism(rhos, ["R"] * 6)
# ValueError: Kinematic parameters length mismatch: expected 7, got 6
# 安检小姐姐："6 个关节只有 6 个旋量参数？您的机器人缺一块骨头。"
```

这意味着你的 bug 会在**第一现场**被抓住，而不是默默污染下游算法，让你在深夜三点对着不对的数值怀疑人生。

### 4. 双层工作流 —— 先涂鸦，再定稿

`lieink` 提供两种画风：

- **🎨 涂鸦模式（Container）**：用 `LieContainer` 做广播、链式调用，快速验证算法逻辑。像草稿纸，涂改方便。
- **📝 定稿模式（cb 方法）**：`Twist.expcb()`、`SE3.invcb()` 等纯 NumPy 批量操作，零额外开销，适合确认后的高效执行。

机构模块遵循同样的哲学：

- **单条调试**：`forward_kinematics` / `coordinate_pose`，一次一个位姿，方便打断点、看中间量；
- **批量执行**：`forward_kinematicsb` / `coordinate_poseb`，成百上千条数据一把梭。

### 5. 为标定而生的关节语义（UMP / MP / A）

每个关节除了运动类型（R/P），还标注**驱动语义**（当前仅 Limb 实现）：

- **A**ctuated：驱动关节（你的输入）；
- **P**assive：被动关节（需要被求解出来的未知量）；
- **M**easurable / **U**nmeasurable：可测 / 不可测（实际零位未知——正是标定要辨识的对象）。

三者自由组合成 `"A"`、`"MP"`、`"UMP"` 等标签，自动派生四张掩码：

```python
self.mask_passive_joints       # 含 "P"
self.mask_actuated_joints      # 含 "A"
self.mask_measurable_joints    # 不含 "U"
self.mask_unmeasurable_joints  # 含 "U"
```

并联机构的闭环求解、约束旋量计算全部基于这套掩码驱动。给 Stewart 平台的支链标上 `["UMP", "UMP", "A", "UMP", "UMP", "UMP"]`——六个转动副是被动的、不可测的（标定的痛），只有中间的移动副是驱动——剩下的交给库。**这是给运动学标定量体裁衣的设计。**

---

## 🚀 快速开始

### 安装

**方式一：直接安装（推荐）**
无需 clone，你可以直接通过 pip 一键安装最新版本：
```bash
pip install git+https://github.com/Huang-Zhenkai/lieink.git
```

**方式二：本地开发安装**
如果你想在本地阅读源码或修改代码，也可以使用 clone 的方式：
```bash
git clone https://github.com/Huang-Zhenkai/lieink.git
cd lieink
pip install -e .
```

> 需要 **Python ≥ 3.12**（我们用了 PEP 695 泛型语法和 `type` 语句，低于 3.12 会直接 `SyntaxError`，没有商量的余地）。
> 依赖：`numpy`, `scipy`, `beartype`, `jaxtyping`, `rich`。

### 最小示例：旋转、点、旋量，一气呵成

```python
import numpy as np
from lieink.atoms import SO3, Point3, Twist3

# 造一个旋转，链式更新，像搭积木一样爽
R = SO3(np.eye(3))
R.rotx(0.1, inplace=True).roty(0.2, inplace=True).rotz(0.3, inplace=True)

# 作用在点和旋量上
# 注意传入 np.array 而非列表
p = Point3(np.array([1, 0, 0]))
t = Twist3(np.array([1, 2, 3]))

(R * p).print()   # 旋转作用在点上 → 得到新点
(R * t).print()   # 旋转作用在旋量上 → 得到新旋量
(R @ t.toso3()).print()  # 伴随作用（@ 运算符）→ 得到 so3
```

### 串联臂：POE 公式直接搬过来，还附赠雅可比

```python
import numpy as np
from lieink.mechanisms.localpoe.serial_mechanisms import SerialMechanism

# POE 参数：N 个关节的初始位姿旋量 ξ + 末端位姿 M，共 N+1 组 6x1
rhos = np.zeros((7, 6, 1))
rhos[:, 5, 0] = 0.1        # 初始位姿：沿 z 平移 0.1
rhos[1:6, 2, 0] = 1.0      # 关节旋量：绕各自局部 z 轴

arm = SerialMechanism(rhos, ["R"] * 6)

# 正运动学 + 空间速度雅可比，一次搞定
pose, J = arm.forward_kinematics(
    ctrl=np.linspace(0.0, 1.0, 6),
    return_vjacobian=True,
    return_NDArray=True,
)

# 还想要每个关节的局部位姿？加一个 return_local_poses=True 就行
```

### Stewart 平台：给铰链，得整机

```python
import numpy as np
from lieink.mechanisms.localpoe.parallel_mechanisms import StewartPlate

angles = np.arange(6) * np.pi / 3
lhs_g = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)])  # 基座铰链 (3, 6)
uhs_e = 0.6 * lhs_g.copy()                                     # 动平台铰链

# 只要铰链几何和初始位姿，6 条 UPS 支链的参数自动算好
stewart = StewartPlate(lhs_g, uhs_e, np.eye(4))

# 封闭逆解：动平台位姿 → 6 根杆长
target = np.eye(4); target[2, 3] = 0.5
ls = stewart.inverse_kinematics(target)

# 协调正解：限制杆长，给定其余关节迭代初始值，Newton 迭代把 6 条支链"劝"到同一个位姿
pose = stewart.coordinate_pose_by_ctrl(ls)
```

### 批量与容错：一百个位姿一起上

```python
# 批量逆解：(100, 4, 4) → 100 组杆长
poses = np.tile(target, (100, 1, 1))
ls_all = stewart.inverse_kinematicsb(poses)

# 批量闭环求解；dont_raise=True 时不抛异常，而是返回逐行收敛掩码
poses_hat, success = stewart.coordinate_pose_by_ctrlb(ls_all, dont_raise=True)
print(f"{success.sum()} / {len(success)} converged")
```

一个值得展开的设计点：批量 Newton 求解时，**已收敛的行直接"毕业"，未收敛的行继续迭代**；实在劝不动的行也不会拖整个 batch 陪葬，而是标记在布尔掩码里交给你定夺——"1000 组数据里有 3 组不收敛"从此不再是噩梦。

同理，`Limb.constraint_wrenches(vjacobian)` 一行给出支链的约束旋量（被动关节速度雅可比的零空间）——并联机构分析里绕不开的那一步，内置。

---

## 🏗️ 架构亮点（一些你可能想知道的幕后故事）

### 运算符注册表：一个开放的"婚介所"

`lieink` 不硬编码所有运算组合，而是通过 `ALLOWED_OPERATORS` 注册表动态管理：

```python
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", Point3, Point3)   # SO3 × Point3 → Point3
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "@", se3, se3)         # 伴随运算
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
| --- | --- | --- |
| `c` | 单例静态计算 | `SO3.rotxc(0.5)` |
| `cb` | 批量静态计算（NumPy 广播） | `SO3.rotxcb(np.array([0.1, 0.2, 0.3]))` |
| 无 | 实例方法（支持 inplace） | `R.rotx(0.5, inplace=True)` |

`cb` 方法内部全是 NumPy 原生操作，没有 Python 层 for 循环，快得飞起。机构模块同样遵循这套命名：`coordinate_pose` / `coordinate_poseb`、`inverse_kinematics` / `inverse_kinematicsb`，看到 `b` 就知道是批量版。

### 机构基类：一层套一层的俄罗斯套娃

抽象基类看着多，但每一层都有明确分工，而且**层层都可被单独继承**（详见下一节的扩展指南）。这不是闲得慌，是为了让你只重写该重写的部分。

---

## 🔧 扩展指南：继承谁，写什么，白嫖什么

弄这么多基类，一大部分原因就是让别人（和三个月后的自己）能优雅地扩展：

| 你想做什么 | 继承谁 | 必须实现 | 免费获得 |
| --- | --- | --- | --- |
| 新串联构型（POE 够用） | `SerialMechanism` | 什么都不用写 | 正解 + 雅可比 + 批量版 |
| 新串联构型（非 POE，自定义正解） | `BasicSerialMechanism` | `forward_kinematics` | 参数合法性校验、`forward_kinematicsb` 批量包装 |
| 新支链类型 | `BasicLimb` | `forward_kinematics` | UMP/MP/A 驱动语义 + 四张掩码；约束旋量可参考 `Limb` 的零空间实现 |
| 新并联拓扑（如 Delta） | `ParallelMechanism` | `inverse_kinematics`（+ `b` 版） | Newton 闭环协调求解、全局掩码管理、`inverse_kinematicsb` |
| 新 Lie 原子类型 | `BasicLie` / `BasicLieAlgebra` / `LieGroup` 家族 | `_check_value`、`logc`/`logcb` 等抽象方法 | 三套 API（c/cb/实例）、安检体系、`UPDATE_ALLOWED_OPERATORS` 登记运算 |
| 新容器 | `BasicContainer` | —— | 广播运算、map、聚合、属性广播 |

举个 🌰：写一个 Delta 机器人，只需要

```python
from lieink.mechanisms.localpoe.parallel_mechanisms import ParallelMechanism

class DeltaPlate(ParallelMechanism):
    def inverse_kinematics(self, pose, only_actuated_joints=True):
        ...  # 你的封闭逆解，返回各支链关节变量
```

闭环协调正解、批量求解、收敛掩码，全部继承而来，一行不用重写。`StewartPlate` 就是这么干的——它只实现了 `inverse_kinematics`，其余全是白嫖。

各基类的分工一览：

- **`BasicSerialMechanism`**：入职体检（关节数与参数数匹配、R/P 类型合法、P 关节不超过 3 个），抽象 `forward_kinematics`；`b` 版由 `batch_func` 自动包装，天然免费。
- **`BasicLimb`**：在串联框架上叠加驱动语义校验，自动生成四张关节掩码。
- **`BasicParallelMechanism[T]`**：泛型容器管理支链；把各支链的掩码缝合成全局掩码；`limbs` setter 内置一致性检查——换支链时数量 / 掩码 / 关节数不符会当场报警，而不是默默算出错误结果。
- **`ParallelMechanism`**：固定支链为 `Limb`，实现基于被动关节的 Newton 闭环求解。
- **`StewartPlate`**：只写逆解，血肉全是继承——这就是骨架设计的意义。

---

## 🆚 和其他库的区别

| 维度 | lieink | 典型李群库（如 sophuspy） | 典型机器人学库 |
| --- | --- | --- | --- |
| **气质** | 研究者的草稿纸 + 钢笔 | 工程师的螺丝刀 | 视库而定 |
| **覆盖范围** | 李群原子 + 串/并联机构整机 | 只给原子，机构自己搭 | 以串联为主，并联支持有限 |
| **参数化** | POE 全局旋量 | —— | 多为 DH/MDH 逐关节局部坐标系 |
| **批量计算** | `b` 系列 + 逐行收敛掩码 | —— | 多数要自己写循环 |
| **关节语义** | UMP/MP/A，为标定设计 | —— | 通常只有 active/passive |
| **类型系统** | 严格一等公民，运行时双重安检 | 常基于裸矩阵或轻量封装 | 视库而定 |
| **可扩展性** | 基类骨架 + 注册表，动态扩展 | 运算规则通常硬编码 | 视库而定 |
| **自动微分** | 不是我们的菜（专注数值正确性） | 部分库支持 JAX/Torch AD | 视库而定 |

几个值得展开的点：

- **POE vs DH**：DH 参数要逐个关节建系，改一处牵一发动全身，标定时还要面对"这个误差到底属于哪个杆件"的灵魂拷问；POE 的每组旋量参数独立、物理意义直白——每个参数都对应一条可辨识的误差，和标定是天作之合。
- **并联机构不是二等公民**：闭环协调求解（`coordinate_pose_by_ctrl`）、约束旋量（`Limb.constraint_wrenches`）、批量容错，全部内置，不用自己写 Newton 迭代写到头秃。
- **批量优先**：从原子的 `cb` 系列到机构的 `b` 系列，批量是一等设计目标，不是事后补丁。

如果你需要的是一个**能让你在半小时内把论文公式变成可运行代码**的工具，`lieink` 很可能是你的菜。

---

## 🎯 适用场景

- 🤖 **机器人运动学与运动学标定研究**：UMP 关节语义、约束旋量、批量容错，为标定量体裁衣
- 🔗 **并联机构研究**：构型综合、闭环求解、性能分析、实验数据批量处理
- 📚 **教学与论文复现**：需要严格对应数学符号、快速试错的研究流程
- 🧪 **算法验证**：先 Container 验证逻辑，再 cb 方法跑实验数据

---

## ⚙️ 环境要求

- **Python ≥ 3.12**（PEP 695 泛型 + `type` 语句；3.11 及以下会在 import 阶段直接 `SyntaxError`）
- 依赖：`numpy`、`scipy`、`beartype`、`jaxtyping`、`rich`

---

## 📜 许可证

[MIT License](LICENSE)

---

> *"数学是墨水，代码是笔，机构是纸上的火柴人——lieink 负责让他们站起来。"*
> >
> *—— 某个研究机器人运动学标定但无人接班的可怜研究生（虽然是根本找不到工作的烂方向）*