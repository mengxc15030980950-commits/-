---
title: SAM学习笔记：argmax扰动如何用一阶近似求解
published: 2026-11-16
description: 从SAM的最小最大目标出发，推导L2约束下的参数扰动闭式解，并补充可运行的PyTorch代码、GRU调用示例和实现验证。
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
- **控制随机性。** 两次使用相同批次，并显式处理随机状态；仅恢复CUDA全局状态并不足以保证所有后端内部状态一致，详见下面的实现说明。

分母中的 $10^{-12}$ 用于避免除零。因此，工程计算得到的扰动范数通常略小于 $\rho$；当梯度恰好为零时，计算结果为零。

### 3.1 可运行的PyTorch实现

下面把伪代码展开为实际函数。`closure()`负责在**同一份已经取出的批次**上计算标量损失；函数内部调用两次，因此不要在`closure`里调用`next(dataloader)`或重新抽样。

示例限定CPU或单张CUDA卡、FP32和稠密梯度，不包含AMP、DDP与BatchNorm运行统计的处理。完整文件可[下载sam-pytorch-example.py](/code/sam-pytorch-example.py)，安装PyTorch后运行`python sam-pytorch-example.py`即可执行合成数据示例。

```python
"""Single-device FP32 SAM example; synthetic data only."""
import math
import torch
from torch import nn


def negative_ic(prediction, target):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = (prediction.square().sum() * target.square().sum()).sqrt()
    return -(prediction * target).sum() / (denominator + 1e-8)


def sam_step(model, optimizer, closure, rho=0.05):
    """closure() computes a scalar loss on the SAME preloaded batch."""
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    params = [p for group in optimizer.param_groups
              for p in group["params"] if p.requires_grad]
    if not params or len({p.device for p in params}) != 1:
        raise ValueError("Use one CPU or CUDA device")
    device = params[0].device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This example supports CPU/CUDA only")
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError("This example uses FP32, without AMP")
    batch_norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.SyncBatchNorm)
    if any(isinstance(module, batch_norm) for module in model.modules()):
        raise ValueError("BatchNorm needs separate running-stat handling")

    model.train()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    # 1. First backward: find the uphill direction.
    optimizer.zero_grad(set_to_none=True)
    loss = closure()
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("The first loss must be a finite scalar")
    loss.backward()
    active = [p for p in params if p.grad is not None]
    if not active:
        raise RuntimeError("No parameter gradients")
    grad_norm = torch.stack([
        p.grad.detach().square().sum() for p in active
    ]).sum().sqrt()
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("Nonfinite first gradient")

    # 2. One global L2 radius; no gradient graph through the perturbation.
    original = [p.detach().clone() for p in active]
    perturbations = [p.grad.detach() * (rho / (grad_norm + 1e-12))
                     for p in active]
    try:
        with torch.no_grad():
            for p, perturbation in zip(active, perturbations):
                p.add_(perturbation)

        # 3. Replay dropout randomness and recompute on the SAME batch.
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        perturbed_loss = closure()
        if perturbed_loss.ndim != 0 or not torch.isfinite(perturbed_loss).item():
            raise FloatingPointError("Nonfinite perturbed loss")
        perturbed_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
               for p in params):
            raise FloatingPointError("Nonfinite second gradient")
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        # copy_ restores the original weights exactly, including on failure.
        with torch.no_grad():
            for p, saved in zip(active, original):
                p.copy_(saved)

    # 4. Keep second-pass gradients; do ONE AdamW/SGD update at original weights.
    optimizer.step()
    perturbation_norm = torch.stack([
        e.square().sum() for e in perturbations
    ]).sum().sqrt()
    return {
        "loss": loss.detach().item(),
        "perturbed_loss": perturbed_loss.detach().item(),
        "gradient_norm": grad_norm.item(),
        "perturbation_norm": perturbation_norm.item(),
    }
```

代码与公式的对应关系如下：

| 代码步骤 | 数学含义／实现作用 |
| --- | --- |
| `loss.backward()` | 计算当前参数处的第一组梯度 |
| `grad_norm` | 对全部参与扰动参数求联合L2范数；不是逐层归一化 |
| `p.add_(perturbation)` | 临时进入扰动参数位置，构造一阶近似的内层最大点 |
| 第二次`zero_grad()`与`backward()` | 清除第一组梯度，只保留扰动位置的第二组梯度 |
| `finally`中的`p.copy_(saved)` | 无论第二次计算是否成功，都恢复精确保存的原参数 |
| `optimizer.step()` | 使用第二组梯度，从原参数执行一次AdamW更新 |

其中`torch.no_grad()`只包住参数扰动和还原，不包住第二次前向计算。两个`backward()`均不启用`create_graph=True`，从而不对扰动构造继续求高阶导数。保存原参数再复制还原，也避免了直接加扰动、再减扰动可能积累的舍入误差。

### 3.2 接到GRU与负IC损失

输入形状为`(批次数, 时间长度, 字段数)`，GRU最后一个时间位置的输出通过线性层生成预测分数。下面沿用两层GRU、隐藏维度64、dropout为0.1的结构，用合成标签演示一次更新：

```python
class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def main():
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GRURegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # x: (batch, time, fields). Synthetic labels only check code execution.
    x = torch.randn(64, 30, 6, device=device)
    y = 0.6 * x[:, -1, 0] - 0.3 * x[:, -1, 1]
    closure = lambda: negative_ic(model(x), y)
    # Native GRU path: avoid cuDNN-owned dropout state in this teaching demo.
    with torch.backends.cudnn.flags(enabled=False):
        stats = sam_step(model, optimizer, closure, rho=0.05)
    print({key: round(value, 6) for key, value in stats.items()})


if __name__ == "__main__":
    main()
```

CPU运行示例输出如下，显示扰动的联合范数约为0.05：

```text
{'loss': 0.082153, 'perturbed_loss': 0.3972, 'gradient_norm': 7.824732, 'perturbation_norm': 0.05}
```

这里的`loss`是更新前的损失，`perturbed_loss`是临时扰动后的损失，二者都不是正式更新后的损失。其他环境的数值可能不同；单步扰动后的真实损失也不保证总比原损失大。一阶近似的性质需要与实际非线性损失区分。

### 3.3 dropout与可复现范围

`sam_step`同时保存和恢复CPU与目标CUDA设备的全局随机状态，可用于重放普通`nn.Dropout`使用的随机数。但这不等于回滚任意算子的全部内部状态；cuDNN RNN可能另有dropout状态。因此，GRU示例在调用时显式关闭cuDNN路径，采用原生PyTorch运算。这会影响性能；若做优化器比较，对照组也应使用相同后端设置。

此前复现代码只保存和恢复CUDA全局随机状态，不能单凭这一步声称已证明cuDNN GRU两次前向的内部掩码完全一致。原有实验代码保持不变；这里补充的是便于理解和验证的独立示例。

### 3.4 已验证哪些行为

该示例在PyTorch 2.9.1的CPU环境上运行，并进行了四项小型检查：

1. 对具有解析解的二次损失，SAM与SGD组合后的参数更新与解析值一致，联合扰动范数约为0.05。
2. 令ρ＝0时，包含普通Dropout的模型与相同随机状态下的一次普通AdamW更新一致，优化器步数为1。
3. 人为让第二次损失计算报错后，模型参数精确恢复，AdamW状态未被更新。
4. 梯度为零时，扰动为零，并且不产生除零错误。

GRU调用示例也已运行。CUDA分支本轮未单独实测；这些检查验证代码行为，不属于股票训练或回测，也不证明SAM能改善量化收益。

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

4. PyTorch文档：[清除梯度](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)、[no_grad](https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html)、[CPU随机状态](https://docs.pytorch.org/docs/2.9/generated/torch.get_rng_state.html)、[CUDA随机状态](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.get_rng_state.html)。
