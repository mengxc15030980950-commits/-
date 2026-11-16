---
title: "AoI 经典论文学习笔记｜Real-Time Status: How Often Should One Update?（2012）"
published: 2026-11-16
description: "信息年龄 AoI 的经典论文笔记：平均 AoI 通用公式、M/M/1、M/D/1、D/M/1 模型比较、Zero-Wait 策略，以及完整附录推导。"
tags: [AoI, 信息年龄, 排队论, 论文笔记]
category: 论文学习
lang: zh_CN
draft: false
---

**论文信息**

- **标题：** Real-Time Status: How Often Should One Update?
- **作者：** Sanjit Kaul、Roy Yates、Marco Gruteser
- **年份：** 2012
- **研究方向：** Age of Information（AoI）、排队论、状态更新系统

本文主要围绕信息年龄（Age of Information，AoI）展开，利用排队论研究如何选择合适的信息更新速率，使接收端获得的信息尽可能新鲜。

笔记重点介绍平均 AoI 的通用公式，以及 **M/M/1、M/D/1、D/M/1** 三种排队模型的计算结果与比较。详细的数学推导统一放在附录中。

---

## 一、研究背景与主要贡献

### 1.1 研究问题

在实时状态更新系统中，信源（Source）生成状态数据包，通过有限服务能力的通信系统发送给接收端（Monitor）。

核心问题是：

**如何选择状态更新速率，使接收端的信息尽可能新鲜？**

更新速率存在以下权衡：

- **更新速率过低：** 数据包生成间隔较长，信息逐渐陈旧。
- **更新速率过高：** 数据包在服务器中排队，系统时延增加。
- **合适的更新速率：** 在信息更新频率与排队时延之间取得平衡。

传统通信系统通常关注吞吐量（Throughput）和时延（Delay），但本文指出，最小化 AoI 与这两个目标并不相同。

### 1.2 主要贡献

1. 建立 FCFS 排队系统中平均 AoI 的通用计算公式。
2. 分析 M/M/1、M/D/1、D/M/1 三类模型的最优更新速率。
3. 研究服务器状态可观测时的 Zero-Wait 更新策略，以及平均 AoI 的下界。

---

## 二、系统模型与 AoI 定义

### 2.1 基本符号

| 符号 | 定义 |
|---|---|
| tᵢ | 第 i 个数据包的生成时刻 |
| t′ᵢ | 第 i 个数据包的接收时刻 |
| Xᵢ | 相邻数据包的生成时间间隔 |
| Wᵢ | 第 i 个数据包的排队等待时间 |
| Sᵢ | 第 i 个数据包的服务时间 |
| Tᵢ | 第 i 个数据包的系统时间 |
| λ | 平均数据包生成速率 |
| μ | 平均服务速率 |
| ρ | 服务器利用率 |

其中：

$$
X_i=t_i-t_{i-1}
$$

$$
T_i=W_i+S_i
$$

$$
\rho=\frac{\lambda}{\mu}
$$

### 2.2 信息年龄（Age of Information）

AoI 定义为：

$$
\boxed{\Delta(t)=t-u(t)}
$$

其中：

- t 表示当前时刻。
- u(t) 表示接收端最近收到的数据包的生成时间。
- Δ(t) 表示当前时刻的信息年龄。

当第 i 个数据包到达时：

$$
\Delta(t'_i)=T_i=t'_i-t_i
$$

在没有新更新到达时，AoI 以斜率 1 线性增长；当新的状态更新到达时，AoI 下降到该数据包的系统时间。

因此，AoI 随时间呈现典型的锯齿形曲线（Sawtooth Pattern）。

### 2.3 平均 AoI 的通用公式

时间平均信息年龄定义为：

$$
\Delta_T=\frac{1}{T}\int_0^T\Delta(t)\,dt
\tag{1}
$$

利用论文 Figure 2 的锯齿面积分解：

$$
Q_i=X_iT_i+\frac{X_i^2}{2}
\tag{4}
$$

在满足相应长期平均收敛条件时，可得到本文最重要的公式：

$$
\boxed{
\Delta=\lambda\left(E[XT]+\frac{E[X^2]}{2}\right)
}
\tag{7}
$$

其中：

- E[X²] 反映更新间隔的二阶统计特征。
- E[XT] 同时考虑更新间隔与系统时延。
- X 与 T 一般存在统计依赖关系，因此不能直接将 E[XT] 拆成 E[X]E[T]。

这一公式是后续三种排队模型分析的共同基础，具体推导见附录 A。

---

## 三、三种排队模型的主要结果

### 3.1 M/M/1：随机到达 + 随机服务

**模型假设**

$$
X_i\sim\mathrm{Exp}(\lambda)
$$

$$
S_i\sim\mathrm{Exp}(\mu)
$$

即数据包按照泊松过程生成，服务时间独立同分布，且服从指数分布。

**关键中间结果**

$$
E[WX]=\frac{\rho}{\mu^2(1-\rho)}
\tag{16}
$$

$$
E[TX]=\frac{\rho}{\mu^2(1-\rho)}
+\frac{1}{\lambda\mu}
$$

**平均 AoI**

$$
\boxed{
\Delta_{\mathrm{M/M/1}}
=\frac{1}{\mu}
\left(
1+\frac{1}{\rho}+\frac{\rho^2}{1-\rho}
\right)
}
\tag{17}
$$

固定服务速率 μ，关于 ρ 求导，并令导数为零：

$$
\rho^4-2\rho^3+\rho^2-2\rho+1=0
$$

数值求解得到：

$$
\boxed{\rho^*_{\mathrm{M/M/1}}\approx0.53}
$$

因此，在最优利用率下：

- 服务器约 53% 的时间处于忙碌状态。
- 服务器约 47% 的时间处于空闲状态。
- 系统中的平均数据包数量约为 1.13。

**核心结论**

泊松到达和指数服务的随机性使高负载下的排队时延显著增加，因此平均 AoI 的最小值并不出现在服务器满负载时。

详细推导见附录 B。

### 3.2 M/D/1：随机到达 + 固定服务

**模型假设**

$$
X_i\sim\mathrm{Exp}(\lambda)
$$

$$
S_i=D=\frac{1}{\mu}
$$

服务时间固定，但数据包仍按照泊松过程到达。

由于：

$$
T_i=W_i+D
$$

得到：

$$
E[T_iX_i]=E[W_iX_i]+DE[X_i]
\tag{18}
$$

**平均 AoI**

$$
\boxed{
\Delta_{\mathrm{M/D/1}}
=\lambda E[WX]+D+\frac{1}{\lambda}
}
$$

其中：

$$
E[WX]=
\int_0^\infty
xE[W_i\mid X_i=x]\lambda e^{-\lambda x}\,dx
$$

M/D/1 的平均等待时间为：

$$
\boxed{
E[W]=\frac{D\rho}{2(1-\rho)}
}
$$

由于 W 与 X 存在依赖关系，无法直接使用 E[W]E[X] 计算 E[WX]。

因此，作者利用等待时间的累积分布函数（CDF）和条件期望，通过数值积分计算 E[WX]，进而优化平均 AoI。

最终得到：

$$
\boxed{\rho^*_{\mathrm{M/D/1}}\approx0.625}
$$

**核心结论**

固定服务时间消除了服务时长的随机波动，使系统能够在更高利用率下维持较小的平均 AoI。

详细推导见附录 C。

### 3.3 D/M/1：固定到达 + 随机服务

**模型假设**

$$
X_i=D
$$

$$
S_i\sim\mathrm{Exp}(\mu)
$$

其中 D 表示固定的数据包生成周期。

因此：

$$
\lambda=\frac{1}{D}
$$

$$
E[X^2]=D^2
$$

由于 X=D 是常数：

$$
E[XT]=DE[T]
$$

将其代入平均 AoI 的通用公式（7）：

$$
\boxed{
\Delta_{\mathrm{D/M/1}}=E[T]+\frac{D}{2}
}
\tag{22}
$$

在 D/M/1 的稳态系统中：

$$
E[T]=\frac{1}{\mu(1-\beta)}
$$

其中 β 满足：

$$
\beta=e^{-\mu D(1-\beta)}
=e^{-(1-\beta)/\rho}
\tag{24}
$$

取稳定系统对应的非平凡解 0 < β < 1。

最终得到：

$$
\boxed{
\Delta_{\mathrm{D/M/1}}
=\frac{1}{\mu}
\left(
\frac{1}{2\rho}+\frac{1}{1-\beta}
\right)
}
\tag{25}
$$

数值优化结果：

$$
\boxed{\rho^*_{\mathrm{D/M/1}}\approx0.515}
$$

**核心结论**

固定更新周期避免了泊松生成过程中长短不均的更新间隔，有助于降低平均 AoI。

详细推导见附录 D。

---

## 四、三种模型的比较与分析

### 4.1 模型对比

| 维度 | M/M/1 | M/D/1 | D/M/1 |
|---|---|---|---|
| 到达间隔 | 指数分布 | 指数分布 | 固定 |
| 服务时间 | 指数分布 | 固定 | 指数分布 |
| 到达随机性 | 有 | 有 | 无 |
| 服务随机性 | 有 | 无 | 有 |
| 最优利用率 | 0.530 | 0.625 | 0.515 |
| AoI 计算方式 | 解析式 | 数值积分 | 隐式方程与数值求解 |

### 4.2 Figure 3：平均 AoI 与利用率的关系

*（此处插入原论文第 4 页 Figure 3：Age Δ 随 Server Utilization ρ 的变化曲线）*

论文 Figure 3 固定服务速率 μ=1，比较三种 FCFS 系统的平均信息年龄随服务器利用率的变化。

**（1）低利用率区域**

当服务器利用率较低时：

- 数据包生成频率较低。
- 服务器通常比较空闲。
- 排队等待时间较短。
- AoI 主要受到更新间隔的影响。

因此，M/M/1 与 M/D/1 的平均 AoI 较为接近。

**（2）高利用率区域**

当服务器利用率较高时：

- 服务器更频繁地出现数据包排队。
- 等待时间对 AoI 的影响逐渐增强。
- M/M/1 中可能出现较长的随机服务时间，导致后续数据包等待更久。
- M/D/1 的固定服务时间可以减少这种影响。

因此，在相同利用率下，M/D/1 的平均 AoI 小于 M/M/1。

**（3）D/M/1 的优势**

D/M/1 通过固定数据包生成间隔，避免了泊松到达过程中的随机间隔波动。

在论文 Figure 3 的比较条件下，D/M/1 的平均 AoI 曲线进一步降低。

这说明，**到达过程和服务过程的随机性都会影响信息新鲜度。**

### 4.3 不同优化目标的区别

| 优化目标 | 对服务器利用率的偏好 |
|---|---|
| 最大化吞吐量 | ρ → 1 |
| 最小化系统时延 | ρ → 0 |
| 最小化平均 AoI | 存在内部最优值 0 < ρ* < 1 |

在稳定的 M/M/1 系统中：

$$
\text{Throughput}=\lambda=\rho\mu
$$

而平均系统时间为：

$$
E[T]=\frac{1}{\mu-\lambda}
=\frac{1}{\mu(1-\rho)}
$$

当 ρ 增大时，吞吐量提高，但平均系统时延也会增大。

因此：

**最小化 AoI 不等于最大化吞吐量，也不等于最小化数据包系统时延。**

状态更新系统需要在更新频率与排队时延之间取得平衡。

---

## 五、Zero-Wait 策略与平均 AoI 下界

第七节进一步考虑信源能够观察服务器状态的情况。

### 5.1 Zero-Wait 更新策略

Zero-Wait 指：当前数据包完成服务的瞬间，立即生成下一个数据包。

因此：

$$
t_i=t'_{i-1}
$$

每个数据包都无需排队：

$$
W_i=0
$$

系统时间等于服务时间：

$$
T_i=S_i
$$

同时：

$$
X_i=S_{i-1}
$$

$$
\lambda=\frac{1}{E[S]}
$$

### 5.2 平均 AoI

在独立同分布服务时间的假设下：

$$
E[XT]=(E[S])^2
$$

$$
E[X^2]=E[S^2]
$$

代入平均 AoI 通用公式，得到：

$$
\boxed{
\Delta^*
=E[S]+\frac{E[S^2]}{2E[S]}
}
\tag{26}
$$

如果服务时间服从参数为 μ 的指数分布：

$$
E[S]=\frac{1}{\mu}
$$

$$
E[S^2]=\frac{2}{\mu^2}
$$

则：

$$
\boxed{\Delta^*=\frac{2}{\mu}}
$$

### 5.3 平均 AoI 下界

由于服务时间方差非负：

$$
\mathrm{Var}(S)=E[S^2]-(E[S])^2\geq0
$$

因此：

$$
E[S^2]\geq(E[S])^2
$$

代入公式（26）：

$$
\Delta^*\geq
E[S]+\frac{(E[S])^2}{2E[S]}
$$

最终得到：

$$
\boxed{
\Delta^*\geq\frac{3E[S]}{2}
=\frac{3}{2\mu}
}
\tag{27}
$$

当服务时间确定时：

$$
\mathrm{Var}(S)=0
$$

上述下界取到等号。

这一结果说明，在平均服务时间固定时，服务时间的随机波动会增加 Zero-Wait 系统的平均 AoI。

这里的下界依赖于论文所考虑的模型与比较范围，不应直接推广为所有可能更新策略的普遍下界。

---

## 六、论文总结与未来研究方向

### 6.1 论文主要结论

本文通过 AoI 指标，研究了有限服务能力下的信息更新速率优化问题。

研究过程可以概括为：

**定义 AoI 指标 → 建立 FCFS 排队模型 → 推导平均 AoI 通用公式 → 分析不同到达与服务分布 → 求解最优利用率 → 探索 Zero-Wait 策略。**

主要结论包括：

**第一，最小化 AoI 与传统的吞吐量最大化、系统时延最小化是不同的优化目标。**

在 FCFS 排队系统中，更新速率过高或过低，都可能导致信息年龄增大。

**第二，到达间隔与服务时间的随机性会影响平均 AoI。**

三种模型得到的最优利用率分别为：

$$
\rho^*_{\mathrm{M/M/1}}\approx0.53
$$

$$
\rho^*_{\mathrm{M/D/1}}\approx0.625
$$

$$
\rho^*_{\mathrm{D/M/1}}\approx0.515
$$

**第三，服务器状态感知能够为进一步降低 AoI 提供可能。**

通过协调数据包生成时间与服务完成时间，Zero-Wait 策略能够消除排队等待，并得到相应的平均 AoI 表达式。

### 6.2 未来研究方向

论文提出了两个主要的扩展方向：

- **不同排队规则：** 例如 LCFS（Last-Come-First-Served，后到先服务）。
- **更复杂的网络模型：** 例如多信源、多服务器以及多个监测端构成的网络。

本文主要研究单信源、单服务器的 FCFS 排队系统，为后续 AoI 更新调度与信息新鲜度优化研究提供了基础。

---

# 附录：核心数学推导

以下部分保留论文主要公式的推导过程，方便复习时查阅。

## 附录 A：平均 AoI 通用公式（7）

定义：

$$
X_i=t_i-t_{i-1}
$$

$$
T_i=t'_i-t_i
$$

根据 Figure 2 的几何关系，梯形面积可以通过两个三角形面积相减得到：

$$
Q_i=\frac{1}{2}(X_i+T_i)^2-\frac{1}{2}T_i^2
$$

展开：

$$
\boxed{
Q_i=X_iT_i+\frac{X_i^2}{2}
}
$$

将观察区间内的面积分解为首尾边界区域及中间梯形区域：

$$
\Delta_T=
\frac{\widetilde Q}{T}
+
\frac{N(T)-1}{T}
\frac{1}{N(T)-1}
\sum_{i=2}^{N(T)}
\left(X_iT_i+\frac{X_i^2}{2}\right)
$$

其中：

$$
\widetilde Q=\widetilde Q_1+\frac{T_n^2}{2}
$$

当 T 趋于无穷大，并满足相应的稳定性与遍历性条件时：

$$
\frac{\widetilde Q}{T}\rightarrow0
$$

$$
\frac{N(T)-1}{T}\rightarrow\lambda
$$

同时，样本平均收敛于对应的数学期望：

$$
\frac{1}{N(T)-1}
\sum_{i=2}^{N(T)}
\left(X_iT_i+\frac{X_i^2}{2}\right)
\rightarrow
E[XT]+\frac{E[X^2]}{2}
$$

因此：

$$
\boxed{
\Delta=\lambda\left(E[XT]+\frac{E[X^2]}{2}\right)
}
\tag{7}
$$

## 附录 B：M/M/1 推导

### B.1 系统时间与等待时间

$$
T_i=W_i+S_i
$$

FCFS 单信源模型满足：

$$
W_i=(T_{i-1}-X_i)^+
$$

其中：

$$
(z)^+=\max(z,0)
$$

在稳态 M/M/1 中，系统时间服从指数分布：

$$
T\sim\mathrm{Exp}(\mu-\lambda)
$$

因此：

$$
f_T(t)=\mu(1-\rho)e^{-\mu(1-\rho)t}
$$

### B.2 条件等待时间

利用 Tᵢ₋₁ 与 Xᵢ 的独立性：

$$
E[W_i\mid X_i=x]=E[(T-x)^+]
$$

因此：

$$
E[W_i\mid X_i=x]
=\int_x^\infty(t-x)f_T(t)\,dt
$$

积分得到：

$$
\boxed{
E[W_i\mid X_i=x]
=\frac{e^{-\mu(1-\rho)x}}{\mu(1-\rho)}
}
$$

### B.3 计算 E[WX]

由于：

$$
f_X(x)=\lambda e^{-\lambda x}
$$

使用全期望公式：

$$
E[WX]
=\int_0^\infty
xE[W_i\mid X_i=x]f_X(x)\,dx
$$

代入：

$$
E[WX]
=\frac{\lambda}{\mu(1-\rho)}
\int_0^\infty xe^{-\mu x}\,dx
$$

利用积分：

$$
\int_0^\infty xe^{-\mu x}\,dx=\frac{1}{\mu^2}
$$

得到：

$$
\boxed{
E[WX]=\frac{\rho}{\mu^2(1-\rho)}
}
$$

### B.4 计算 E[TX]

因为：

$$
E[TX]=E[WX]+E[S]E[X]
$$

且：

$$
E[S]E[X]=\frac{1}{\mu\lambda}
$$

所以：

$$
E[TX]
=\frac{\rho}{\mu^2(1-\rho)}
+\frac{1}{\mu\lambda}
$$

由于指数分布满足：

$$
E[X^2]=\frac{2}{\lambda^2}
$$

代入平均 AoI 公式：

$$
\Delta=
\lambda\left[
\frac{\rho}{\mu^2(1-\rho)}
+\frac{1}{\mu\lambda}
+\frac{1}{\lambda^2}
\right]
$$

使用 λ=ρμ，整理得到：

$$
\boxed{
\Delta=\frac{1}{\mu}
\left(1+\frac{1}{\rho}+\frac{\rho^2}{1-\rho}\right)
}
$$

### B.5 求最优利用率

对 ρ 求导：

$$
\frac{d\Delta}{d\rho}
=\frac{1}{\mu}
\left[
-\frac{1}{\rho^2}
+\frac{2\rho-\rho^2}{(1-\rho)^2}
\right]
$$

令导数为零：

$$
\rho^4-2\rho^3+\rho^2-2\rho+1=0
$$

数值求解：

$$
\boxed{\rho^*\approx0.53}
$$

---

## 附录 C：M/D/1 推导

### C.1 固定服务时间

$$
S_i=D
$$

$$
\rho=\lambda D
$$

因此：

$$
T_i=W_i+D
$$

$$
E[T_iX_i]=E[W_iX_i]+DE[X_i]
$$

代入公式（7）：

$$
\boxed{
\Delta=\lambda E[WX]+D+\frac{1}{\lambda}
}
$$

### C.2 条件等待时间

根据 FCFS 等待关系：

$$
W_i=(W_{i-1}+D-X_i)^+
$$

因此：

$$
E[W_i\mid X_i=x]
=E[(W+D-x)^+]
\tag{19}
$$

M/D/1 的平均等待时间：

$$
E[W]=\frac{D\rho}{2(1-\rho)}
$$

当 x≤D 时：

$$
E[W_i\mid X_i=x]=E[W]+D-x
$$

当 x>D 时，需要考虑 W 的完整概率分布。

等价地，可以利用尾概率表示：

$$
E[W_i\mid X_i=x]
=\int_{x-D}^\infty(1-F_W(w))\,dw
$$

### C.3 利用 CDF 数值计算

论文给出的等待时间累积分布函数为：

$$
F_W(w)=(1-\rho)
\sum_{k=0}^{\lfloor w/D\rfloor}
\frac{\rho^k}{k!}
\left(k-\frac{w}{D}\right)^k
e^{\rho(w/D-k)}
\tag{21}
$$

其中：

$$
\left\lfloor\frac{w}{D}\right\rfloor
$$

表示向下取整。

将其用于计算条件等待时间，再代入：

$$
E[WX]=
\int_0^\infty
xE[W_i\mid X_i=x]\lambda e^{-\lambda x}\,dx
$$

得到：

$$
\Delta(\rho)=\lambda E[WX]+D+\frac{1}{\lambda}
$$

作者通过数值计算寻找使 Δ 最小的利用率：

$$
\boxed{\rho^*\approx0.625}
$$

---

## 附录 D：D/M/1 推导

### D.1 固定到达间隔

$$
X_i=D
$$

$$
\lambda=\frac{1}{D}
$$

因此：

$$
E[X^2]=D^2
$$

$$
E[XT]=DE[T]
$$

代入公式（7）：

$$
\Delta=\frac{1}{D}
\left(DE[T]+\frac{D^2}{2}\right)
$$

得到：

$$
\boxed{
\Delta=E[T]+\frac{D}{2}
}
\tag{22}
$$

### D.2 求解平均系统时间

D/M/1 的稳态排队论结果：

$$
E[T]=\frac{1}{\mu}+\frac{\beta}{\mu(1-\beta)}
$$

整理：

$$
\boxed{
E[T]=\frac{1}{\mu(1-\beta)}
}
$$

其中 β 满足：

$$
\beta=L_X(\mu(1-\beta))
$$

Lₓ(s) 表示到达间隔分布的拉普拉斯变换。

由于 X=D：

$$
L_X(s)=e^{-sD}
$$

因此：

$$
\beta=e^{-\mu(1-\beta)D}
$$

利用：

$$
\rho=\frac{1}{\mu D}
$$

得到：

$$
\boxed{
\beta=e^{-(1-\beta)/\rho}
}
$$

### D.3 平均 AoI 与最优利用率

将 E[T] 代入公式（22）：

$$
\Delta=\frac{D}{2}+\frac{1}{\mu(1-\beta)}
$$

由于：

$$
D=\frac{1}{\mu\rho}
$$

最终：

$$
\boxed{
\Delta=\frac{1}{\mu}
\left(\frac{1}{2\rho}+\frac{1}{1-\beta}\right)
}
\tag{25}
$$

联立 β 与 ρ 的关系，通过数值优化：

$$
\boxed{\rho^*\approx0.515}
$$

---

## 附录 E：Zero-Wait 与 AoI 下界

Zero-Wait 满足：

$$
t_i=t'_{i-1}
$$

$$
W_i=0
$$

因此：

$$
X_i=S_{i-1}
$$

$$
T_i=S_i
$$

由于各服务时间独立同分布：

$$
E[XT]=E[S_{i-1}S_i]=(E[S])^2
$$

同时：

$$
\lambda=\frac{1}{E[S]}
$$

$$
E[X^2]=E[S^2]
$$

代入公式（7）：

$$
\boxed{
\Delta^*=E[S]+\frac{E[S^2]}{2E[S]}
}
\tag{26}
$$

根据方差公式：

$$
\mathrm{Var}(S)=E[S^2]-(E[S])^2
$$

可以改写成：

$$
\boxed{
\Delta^*
=\frac{3E[S]}{2}
+\frac{\mathrm{Var}(S)}{2E[S]}
}
$$

因为：

$$
\mathrm{Var}(S)\geq0
$$

所以：

$$
\boxed{
\Delta^*\geq\frac{3E[S]}{2}
=\frac{3}{2\mu}
}
\tag{27}
$$

当服务时间确定时，方差为零，下界取到等号。

---

## 参考文献

Kaul, S., Yates, R., & Gruteser, M. (2012). *Real-Time Status: How Often Should One Update?*

---

**全文总结**

本文最重要的数学基础是平均 AoI 通用公式：

$$
\boxed{\Delta=\lambda\left(E[XT]+\frac{E[X^2]}{2}\right)}
$$

在此基础上，论文研究了不同到达过程与服务时间分布对信息新鲜度的影响，分别得到 M/M/1、M/D/1 和 D/M/1 的最优服务器利用率，并进一步讨论了 Zero-Wait 策略与平均 AoI 下界。

**最核心的认识是：信息更新并不是越频繁越好，真正需要优化的是接收端的信息新鲜度，而不是单纯的数据包发送数量或传输时延。**

