---
title: SAM学习笔记：argmax扰动如何用一阶近似求解
published: 2026-11-16
description: 从SAM的最小最大目标出发，推导L2约束下的参数扰动闭式解，并说明两次反向传播、AdamW更新及一阶近似的边界。
tags: [机器学习, SAM, 优化算法, 量化研究]
category: 论文笔记
lang: zh-CN
draft: false
author: cheng
sourceLink: https://arxiv.org/abs/2010.01412
---

SAM训练中的一个关键问题是：怎样找到当前参数附近、让损失尽可能大的扰动？本篇围绕这个问题整理目标、推导和实现，区分原始非线性问题与实际训练采用的近似。

论文为Foret、Kleiner、Mobahi和Neyshabur的 *Sharpness-Aware Minimization for Efficiently Improving Generalization*。其第2节给出训练目标和算法。下面的数学推导以常用的L2参数邻域为例；量化损失与实现参数单独说明。

## 1. SAM优化什么

设模型参数为 $w$，一个训练批次为 $B$，对应损失为 $L_B(w)$。SAM考察的邻域损失是

$$
F_B(w)=\max_{\|\epsilon\|_2\leq\rho}L_B(w+\epsilon),
$$

其中 $\rho$ 是参数扰动半径。外层希望减小这个邻域内的最大损失。原论文还在外层目标中加入权重正则项。这里先关注内层扰动；本轮工程实现使用AdamW处理权重衰减。

直观上，若参数移动很短的距离就使损失大幅上升，邻域目标就会惩罚这种位置。这个动机不能直接推出量化策略一定获得更高的样本外收益。[原论文，第2节](https://arxiv.org/html/2010.01412v3#S2)

需要求解的原始扰动为

$$
\epsilon^\star(w)
\in\operatorname*{arg\,max}_{\|\epsilon\|_2\leq\rho}
L_B(w+\epsilon).
$$

这里用 $\in$，是因为最大点可能不唯一。神经网络损失通常是非线性的，实际训练中不会精确搜索这个球内的全部参数位置。

## 2. 先线性化，再求闭式解

记当前梯度为

$$
g=\nabla_w L_B(w).
$$

在 $w$ 附近做一阶展开：

$$
L_B(w+\epsilon)
\approx L_B(w)+g^\top\epsilon.
$$

与 $\epsilon$ 无关的常数项可以去掉，于是得到一个更简单的问题：

$$
\max_{\|\epsilon\|_2\leq\rho}g^\top\epsilon.
$$

下面直接推导其解。由柯西不等式，

$$
g^\top\epsilon
\leq\|g\|_2\|\epsilon\|_2
\leq\rho\|g\|_2.
$$

若 $g\neq0$，取

$$
\widehat\epsilon
=\rho\frac{g}{\|g\|_2},
$$

就有 $\|\widehat\epsilon\|_2=\rho$，并且

$$
g^\top\widehat\epsilon=\rho\|g\|_2.
$$

它达到了上界，因此是**线性化问题的精确最优解**。将其用于原来的非线性损失时，它仍然只是近似扰动。这正是原论文公式(2)在L2约束下的形式。[原论文，公式(2)](https://arxiv.org/html/2010.01412v3#S2)

### 一个二维例子

若 $g=(3,4)$，则 $\|g\|_2=5$。取 $\rho=0.05$，有

$$
\widehat\epsilon=(0.03,0.04),\qquad
\|\widehat\epsilon\|_2=0.05.
$$

线性化模型中的损失增量为

$$
g^\top\widehat\epsilon
=3\times0.03+4\times0.04
=0.25.
$$

这个0.25是线性预测值，实际损失的增量还取决于高阶项。

## 3. 两次反向传播怎样连接起来

得到了扰动以后，还需要决定怎样更新原参数。标准的一阶SAM流程在扰动位置计算第二次梯度，并将该梯度交给基础优化器。官方实现及训练说明可参见[作者代码仓库](https://github.com/google-research/sam)。

下面是更新顺序的伪代码：

```python
# 在当前参数上计算梯度
original_weights = copy_weights(model)
g = gradient(loss_on_batch(model, batch))

# 使用全模型梯度的联合L2范数
perturbation = rho * g / (global_l2_norm(g) + 1e-12)
set_weights(model, original_weights + perturbation)

# 同一批数据，在扰动位置重新计算梯度
clear_gradients(model)
g_sam = gradient(loss_on_batch(model, batch))

# 恢复原参数，再用第二次梯度进行一次正式更新
set_weights(model, original_weights)
optimizer.step_with_gradient(g_sam)
```

正式更新只发生一次。第一次梯度用于寻找增大损失的方向，第二次梯度用于更新原参数。若基础优化器为普通SGD，更新可写为

$$
w_{t+1}=w_t-\eta\nabla L_B(w_t+\widehat\epsilon_t).
$$

使用AdamW时，第二次梯度进入AdamW的动量、二阶矩和权重衰减计算，不能把整个更新简单写成上述SGD形式。

实现时还需要注意三个细节：

- **联合归一化。** 分母是所有参与扰动参数的梯度联合范数，各层共享一个半径。若每层分别归一化，就改变了约束集合。
- **恢复参数。** 第二次梯度算完以后先还原权重，再执行优化器更新。
- **控制随机性。** 本轮实现两次使用相同批次，并恢复相同的CUDA随机状态，使GRU的dropout掩码保持一致。

分母中的 $10^{-12}$ 用于避免除零。因此，工程计算得到的扰动范数通常略小于 $\rho$；当梯度恰好为零时，计算结果为零。

## 4. 实际上用了两层近似

第一层是前面的一阶线性化：用 $\widehat\epsilon$ 代替原非线性问题的最大点。

第二层发生在外层求导。把 $L_B(w+\widehat\epsilon(w))$ 当作一个复合函数完整求导，会出现扰动对参数的导数：

$$
\nabla_w L_B(w+\widehat\epsilon(w))
=
\bigl(I+D_w\widehat\epsilon(w)\bigr)^\top
\left.\nabla_u L_B(u)\right|_{u=w+\widehat\epsilon(w)}.
$$

其中 $D_w\widehat\epsilon(w)$ 是扰动的雅可比矩阵。常用的一阶实现忽略这一项，只采用扰动位置的损失梯度。原论文公式(3)明确给出了这个近似。代码中通过不跟踪扰动构造的计算图来实现，因而不会对第一次梯度继续做高阶求导。[原论文，公式(3)](https://arxiv.org/html/2010.01412v3#S2)

### 零梯度不代表邻域损失不增加

考虑一维函数 $L(w)=w^2$，在 $w=0$ 时梯度为零，一阶扰动会给出零。但真实邻域最大损失为

$$
\max_{|\epsilon|\leq\rho}\epsilon^2=\rho^2,
$$

最大点在 $\epsilon=\rho$ 或 $\epsilon=-\rho$。这个例子说明，驻点附近的曲率可能主导邻域损失；单步一阶近似不能保证找到真实最坏点。

## 5. 放到量化模型中怎样理解

本轮量化实现把损失设为批次内预测分数与中性化收益标签的负Pearson相关系数：

$$
L_B(w)=-\operatorname{Corr}\bigl(f_w(X_B),Y_B\bigr).
$$

因此，内层沿着让负IC上升的方向扰动，也就是局部降低预测相关性的方向。外层再利用这个扰动位置的梯度更新原模型。本轮使用 $\rho=0.05$、AdamW作为基础优化器；这是已选定的实验配置，不能解释为对所有股票池和时间段都适用的参数。

量化评估还需要把以下问题分别核验：

| 问题 | 对应证据 |
| --- | --- |
| SAM更新是否按预定流程执行 | 两次反传、参数还原、梯度归一化与更新日志 |
| 训练是否充分 | 训练和验证曲线、更新预算对照、跨种子稳定性 |
| 预测是否有样本外排序能力 | 按日截面IC、RankIC及分期表现 |
| 信号能否转化为组合超额 | 同股票池、约束、换手及费用下的回测 |

本文只整理算法与实现。尚待核实的数据异常和研究收益不作为这里的结论，也不根据测试收益反向选择扰动半径。关于参数邻域的优化动机、实际预测能力和可交易收益，需要分别提供证据。

## 参考资料

1. Foret, P., Kleiner, A., Mobahi, H., Neyshabur, B. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021：[论文](https://arxiv.org/abs/2010.01412)，重点为第2节、公式(1)—(3)及算法1。
2. [作者代码仓库：google-research/sam](https://github.com/google-research/sam)。仓库使用JAX；本文伪代码用于解释本轮PyTorch实现的更新顺序。
3. 研究背景：华泰证券，2024年10月10日，《SAM，提升AI量化模型的泛化性能》。上述闭式解推导依据SAM原论文；本轮量化参数属于独立复现配置。
