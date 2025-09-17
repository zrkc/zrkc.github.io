---
title: Online Convex Optimization Warmup
date: 2025-09-06 11:01:07
tags: [OCO, Tutorial]
excerpt: Learn to make a trade-off between the past and the present.
---

> 本文是为[高级优化课程](https://www.pengzhao-ml.com/course/AOpt2025fall/)的学生在学习在线凸优化（Lecture 5）之前准备的。具体来说，本文从优化角度引入在线凸优化，并对于在线凸优化的一个经典 setting，介绍在线梯度下降算法和其理论保障。  
> 本文的目的是希望读者能在开启正式学习之前，先对其理论推导有个大致的概览、并做一点思考探究甚至是怀疑，以免在后续学习过程中陷于一步步数学证明的验证而失去了最宝贵的兴趣。为照顾逻辑表达，本文行文叙述和理论推导可能不会十分严谨。欢迎评论！:)

## 0. Introduction: Why Online Optimization?

本文从传统的优化角度引入，建议至少看完第 $0$ 节后再决定是否继续看下去 :)

### 0.1 (Offline) Optimization

大多数人接触优化是从随机梯度下降（Stochastic Gradient Descent, SGD）开始的，再后来就是调包创建优化器：
```py
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
并训练（优化）：
```py
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```
截止目前（2025年），最常用的优化器是 [Adam](https://arxiv.org/abs/1412.6980) 以及其变体 [AdamW](https://arxiv.org/abs/1711.05101)。优化器还有很多，但是为什么现在大家几乎默认使用 Adam 呢？一个很有意思的[观点](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)是：Adam 很适合优化若干年前的神经网络，因此神经网络近些年都在朝着能够继续使得 Adam 有效甚至越来越有效的方向进化。至于在此期间还有其他可能的新架构，往往因为诞生之初难以被*默认*的 Adam 优化而被忽视了。如果事实真就如此，我们能否从理论上说明这个现象，并指导下一个优化器（或者更可能说是，下一对“模型-优化器”）的设计呢？

我们需要回到优化的数学形式化描述上。例如，对于“feature-target”对的数据集 $\\{(\mathbf{z} _n,\mathbf{y} _n)\\} _{n=1}^N$，我们定义参数为 $\mathbf{w}$ 的模型为函数 $\mathcal{M}(\cdot;\mathbf{w}):\mathbf{z}\mapsto\mathcal{M}(\mathbf{z};\mathbf{w})$。定义单值的损失函数 $l(\cdot,\cdot)$ 用以衡量模型输出 $\mathcal{M}(\mathbf{z} _n;\mathbf{w})$ 和真实目标 $\mathbf{y} _n$ 之间的差距。我们设定 $\mathbf{w}$ 属于某个参数空间的集合 $\mathcal{W}\subseteq\mathbb{R}^d$，那么优化问题可以写成：

$$
\begin{equation}
    \min _{\mathbf{w}\in\mathcal{W}}\quad  \frac{1}{N} \sum _{n=1}^N l(\mathcal{M}(\mathbf{z} _n;\mathbf{w}),\mathbf{y} _n). \tag{1}
\end{equation}
$$

我们假设数据集是从某个分布 $\mathcal{D}$ 上独立同分布采样得到的，从而令优化问题为：

$$
\begin{equation}
    \min _{\mathbf{w}\in\mathcal{W}}\quad  \mathbb{E} _{(\mathbf{z,y})\sim\mathcal{D}}\left[ l(\mathcal{M}(\mathbf{z};\mathbf{w}),\mathbf{y})\right]. \tag{2}
\end{equation}
$$

上述问题可以更一般化：我们定义优化目标函数为 $\mathcal{L}(\mathbf{w}):\mathbf{w}\mapsto \mathbb{R}$，并假设它有一个随机 Oracle $\ell(\mathbf{w};\xi)$，满足 $\mathbb{E}_\xi[\ell(\mathbf{w};\xi)\mid\mathbf{w}] = \mathcal{L}(\mathbf{w})$。可以[证明](https://optmlclass.github.io/notes/optforml_notes.pdf)此时还有 $\mathbb{E}_\xi[\nabla\ell(\mathbf{w};\xi)\mid\mathbf{w}] = \nabla\mathcal{L}(\mathbf{w})$。例如，对于某个样本 batch $\mathcal{B}=\\{(\mathbf{z} _n,\mathbf{y} _n)\\}$，对应的损失 $\frac{1}{|\mathcal{B}|}\sum _{(\mathbf{z} _n,\mathbf{y} _n)\in\mathcal{B}}l(\mathcal{M}(\mathbf{z} _n;\mathbf{w}),\mathbf{y} _n)$ 可以看作是对随机 Oracle 的一次访问 $\ell(\mathbf{w};\mathcal{B})$。最终优化问题为：

$$
\begin{equation}
    \min _{\mathbf{w}\in\mathcal{W}}\quad \mathcal{L}(\mathbf{w}). \tag{3}
\end{equation}
$$

这个优化问题也被称为“随机优化（Stochastic Optimization）”。我们将最优参数记作 $\mathbf{w}^\star\triangleq \arg\min _{\mathbf{w}\in\mathcal{W}}\mathcal{L}(\mathbf{w})$。

这里我们研究优化过程的收敛性质。具体来说，假设优化过程中我们访问了 $T$ 次随机 Oracle（例如，计算了 $T$ 个 batch），优化器最终给出的模型参数为 $\mathbf{w}^\dagger$。考虑 $\mathbf{w}^\dagger$ 的损失和最优参数 $\mathbf{w}^\star$ 的差距（optimality gap），（渐进意义下）是否存在某个关于 $T$ 的函数作为上界——比如说，关于 Oracle 随机性的期望上界：<div id="optimality-gap"></div>

$$
\begin{equation}
    \mathbb{E}\left[ \mathcal{L}(\mathbf{w}^\dagger) - \mathcal{L}(\mathbf{w}^\star) \right] \le \epsilon(T), \tag{4}
\end{equation}
$$

又或者高概率上界，即以至少 $(1-\delta)$ 的概率：

$$
\begin{equation}
\mathcal{L}(\mathbf{w}^\dagger) - \mathcal{L}(\mathbf{w}^\star) \le \epsilon(T,\delta). \tag{5}
\end{equation}
$$

正常来说我们期望 $\epsilon(T)$ 是随着 $T$ 递减的，常见的形式比如 $\mathcal{O}\big(\frac{1}{\sqrt{T}}\big),\mathcal{O}\big(\frac{1}{T}\big)$ 等。

### 0.2 Offline to Online

讲到这里，和我们要介绍的在线优化是什么关系呢？我们考虑模型的两个阶段：离线训练和在线部署。离线阶段，模型对着固定的数据集训练；而在线部署后，模型面对流式到来的数据，也可能有优化参数以适应新数据的需要——这方面的例子比如后训练（或者多次后训练）、on-policy 强化学习、持续学习等。为了对比和区分，我们将前者的优化过程（例如上文提到的随机优化）称为“离线优化”（Offline Optimization），而将后者称为“在线优化”（Online Optimization）。你可能会问，随机优化里一个个 batch 的数据，不也是流式的吗？——确实如此，这也是在线学习方法可以很自然地迁移到离线优化的一个原因——但离线和在线更重要的区别在于：**数据分布是否会发生变化**。

对于离线优化而言，优化过程中 Oracle 的分布始终不变，就比如在随机优化中，每个 batch 都是**独立同分布**的。事实上，当数据集被扩充、或者进入在线部署阶段后，数据分布很可能会发生变化，始终独立同分布的条件就不成立了。此时，优化问题从“离线”的一成不变转为“在线”的持续变化，“在线优化”也就有了意义。

等等，在上一小节我们不是说好了要研究优化器的理论吗？这饼给我画哪去了？事实上，正如上所述，在线优化的方法可以很自然地用于离线优化尤其是随机优化，原因也很简单：在线学习研究的是分布可以发生变化，它的一个特殊情况就是分布不变的随机优化。而从理论上，也有一个很简单直接的转化，称为[“online-to-batch conversion”](https://parameterfree.com/2019/09/17/more-online-to-batch-examples-and-strong-convexity/)，它的意思是通过为离线优化设计一种黑盒调用在线优化算法的框架，式 [$(4)$](#optimality-gap) 中的 optimality gap 可以被平均意义下的在线优化指标——遗憾（Regret）——给 upper bound 住，从而我们的问题转变为了为 Regret 提供理论保障。

值得注意的是，现在风生水起的 [Adam (2014)](https://arxiv.org/abs/1412.6980) 优化器，在论文中就是从在线凸优化角度给出收敛性证明的（虽然后续被指出证明存在错误...存在反例使得 Adam 不收敛...）；更值得注意的是，Adam 算法的设计结合了 1) [AdaGrad (2010)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36483.pdf) 的变体 [RMSProp (2014)](https://arxiv.org/pdf/1308.0850v5.pdf) 和 2) 动量方法（早已有之，而最重要的工作是 Yurii Nesterov 在 1983 年提出的 [Nesterov Accelerated Gradient](https://link.springer.com/book/10.1007/978-3-319-91578-4)，大幅加速了梯度下降的收敛率）。AdaGrad 作为在线凸优化和随机优化算法，将在线凸优化中一个很重要的步长设计 [adaptive learning rate (2010)](https://arxiv.org/abs/1002.4862)（本文后面会介绍）直接迁移到 entry-wise，从而更好地利用梯度的稀疏性。总之，Adam(W) 上溯祖宗三代都有在线凸优化——虽然它如今已经成为了大模型时代的默认优化器（而且还是**非凸**的...）——是因为 Adam 恰好成为了大模型时代爆发前夜被选中的[“幸运儿”](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)（那是否可以从在线凸优化诞生出新的幸运儿呢？），亦或是说在线凸优化本就有着超出我们当前认知的潜力呢？

在线优化所属的在线学习（Online Learning），最早由[《Prediction, learning, and games》](https://www.researchgate.net/profile/Gabor-Lugosi/publication/220690817_Prediction_Learning_and_Games/links/0912f50eae7fc7be04000000/Prediction-Learning-and-Games.pdf)一书从博弈论的角度所形式化，并给出了 Regret 概念作为理论指标。当然，作为研究分布变化情况下的优化问题，在线优化还具备更广泛的意义和应用场景。没有一成不变的环境，为了得到有持续适应能力的模型，在线优化也算是起了一个开头了。

## 1. Problem Formulation

在线优化属于在线学习（Online Learning）。在线学习简单来说就是设计与环境交互的算法，根据得到的反馈不断更新自身模型以适应。我们将这个过程抽象成学习者与环境的**博弈**（实际上“博弈”这个词有一定误导性，因为环境不一定总是对抗性的），即如下范式：

<div style="border: 1px solid rgba(127, 127, 127, 1); padding: 10px; border-radius: 5px;">

**在线学习.**  
已知决策可行域 $\mathcal{X}\subseteq\mathbb{R}^d$。在第 $t=1,2,\dots$ 轮：
1. 学习者提交决策 $\mathbf{x} _t\in\mathcal{X}$；
2. 环境选择在线函数 $f _t:\mathcal{X}\to\mathbb{R}$；
3. 学习者受到 $f _t(\mathbf{x} _t)$ 的损失，观测到某些 $f _t$ 的信息并更新决策。
</div>

在线凸优化（Online Convex Optimization, OCO）研究可行域 $\mathcal{X}$ 为凸集且在线函数 $f _t$ 均为凸函数的情况。

### 1.1 Performance Measure

博弈双方的指标是什么呢？最直接地，我们希望学习者能最小化累积损失：

$$
\def \x {\mathbf{x}}
\sum _{t=1}^T f _t(\x _t). \tag{6}
$$

但这个指标对于学习者来说很不公平，因为环境可以对抗性地根据 $\mathbf{x} _t$ 给出 $f _t$，导致很难为该指标给出有意义的保障。类似式 [$(4)$](#optimality-gap) 的 optimality gap（也称为超额风险）引入最优参数作为 comparator，这里我们引入**全局最优固定决策** $\mathbf{x} _\star\triangleq \arg\min _{\mathbf{x}\in\mathcal{X}}\sum _{t=1}^T f _t(\mathbf{x})$ 作为学习者的 comparator，得到称为“遗憾”（Regret）的指标：<div id="regret"></div>

$$
\def \x {\mathbf{x}}
Reg _T \triangleq \sum _{t=1}^T f _t(\x _t) - \min _{\x\in\mathcal{X}}\sum _{t=1}^T f _t(\x). \tag{7}
$$

学习者的目标是最小化遗憾。由于环境的决策此时也会影响 $\mathbf{x} _\star$ 和其损失，因此这是一个更合理的博弈指标。

到这里，我们也可以解释一下上文提到的[“online-to-batch conversion”](https://parameterfree.com/2019/09/17/more-online-to-batch-examples-and-strong-convexity/)、即将随机优化转化为在线优化是怎么一回事了。在随机优化中，假设一共访问了 $T$ 次随机 Oracle $\ell(\mathbf{x};\xi _t),t\in[T]$，其中 $\xi _t$ 是第 $t$ 个 batch 代表的随机变量。我们将它迁移到在线学习框架中：定义第 $t$ 轮的在线函数 $f _t(\mathbf{x})\triangleq \ell(\mathbf{x};\xi _t)$，然后将随机优化最终输出的参数定义为 $\mathbf{w}^\dagger\triangleq \frac{1}{T}\sum _{t=1}^T \mathbf{x} _t$。可以[证明](https://parameterfree.com/2019/09/17/more-online-to-batch-examples-and-strong-convexity/)，当目标函数 $\mathcal{L}(\mathbf{x})$ 为凸函数时，有：<div id="online-to-batch"></div>

$$
\def \x {\mathbf{x}}
\mathbb{E}\left[\mathcal{L}(\mathbf{w}^\dagger) - \mathcal{L}(\mathbf{w}^\star)\right] = \mathbb{E}\left[\mathcal{L}\left( \frac{1}{T}\sum _{t=1}^T \mathbf{x} _t \right) - \mathcal{L}(\mathbf{w}^\star)\right] \le \frac{1}{T}\sum _{t=1}^T \left(\ell(\x _t;\xi_t) - \ell(\mathbf{w}^ \star;\xi _t)\right) \le \frac{Reg _T}{T}, \tag{8}
$$

其中 $Reg _T$ 的定义见 [$(7)$](#regret)。这个结论也比较好理解，既然 $\xi_1,\dots,\xi_T$ 是独立同分布的，那么每一轮的在线函数 $f _t(\mathbf{x})\triangleq \ell(\mathbf{x};\xi _t)$ 期望上就是目标函数 $\mathcal{L}(\mathbf{x})$，Regret 的平均就和 optimality gap 挂钩了。

事实上，Regret 的 comparator 往往并不一定局限在全局固定最优解，而可以是可行域内的**任意**一点。这给了 Regret 非常大的普适性，因此不一定局限于处理流式数据这样的具体问题：凡是能抽象成序列化的问题，都可以尝试归约到 Regret 上。例如，如果你的分析中有一项是 $\sum _{t=1}^T f _t(\mathbf{x} _t)$，可以考虑引入一个**虚拟的** comparator $\mathbf{u}$（或者 comparator 序列）：

$$
\def \x {\mathbf{x}}
\sum _{t=1}^T f _t(\x _t) = \underbrace{\sum _{t=1}^T (f _t(\x _t) - f _t(\mathbf{u}))} _{Regret} + \sum _{t=1}^T f _t(\mathbf{u}).
$$

对于等号右边的 Regret，我们可以套用在线学习的结论，而对于 $\mathbf{u}$ 我们可以根据所需任意选择，甚至将这一项挪到不等号的另一边。总之，这给了我们的分析框架很大的自由度。

### 1.2 Beginning with A Trivial Linear Bound

本小节我们做一个简单的 Regret 上界分析。在本文中，我们还同时关注两个 OCO 中比较常见的假设：可行域有界假设、梯度有界假设。

<div style="border: 1px solid rgba(127, 127, 127, 1); padding: 10px; border-radius: 5px;">

- **可行域有界假设.** 对任意 $\mathbf{x,y}\in\mathcal{X}$，有 $\Vert\mathbf{x-y}\Vert _2\le D$.  
- **梯度有界假设.** 对任意 $\mathbf{x}\in\mathcal{X}$ 和 $t\in[T]$，有 $\Vert\nabla f _t(\mathbf{x})\Vert _2\le G$。即在线函数是 $G$-Lipschitz 的。

</div>

下文简记 $\ell _2$-范数 $\Vert\cdot\Vert _2$ 为 $\Vert\cdot\Vert$。引入这两个假设最直接的作用就是简化了算法设计和证明。这两个假设也很基本，从量纲上来说分别对应了距离和斜率，乘起来就是损失。当然也有很多工作在研究不依赖这两个假设或者对应参数的算法，至今仍有很多有待研究的空间。

现在我们考虑第一个问题：什么样的 Regret 上界是好的呢？你可能会注意到，式 [$(8)$](#online-to-batch) 暗示了一个最基本的要求：Regret 不能是 $\mathcal{O}(T)$ 甚至更大的，不然对应的随机优化算法就没有收敛保证了。事实上，$\mathcal{O}(T)$ 的 Regret 是非常 trivial 的，甚至对于任意决策序列 $\\{\mathbf{x} _t \\} _{t=1}^T$ 都成立。根据微分中值定理：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\def \c {\mathbf{c}}
\begin{align*}
Reg _T &= \sum _{t=1}^T f _t(\x _t) - \sum _{t=1}^T f _t(\xs) = \sum _{t=1}^T \langle \nabla f _t(\c _t), \x _t - \xs \rangle \tag*{$(\c _t\in[\x _t,\xs])$} \\\\
&\le \sum _{t=1}^T \Vert \nabla f _t(\c _t) \Vert \Vert\x _t - \xs\Vert \le GDT.
\end{align*}
$$
其中第一个不等号使用了 Hölder 不等式。我们可以稍微观察一下这个上界：$GD$ 是最大“斜率”乘最大“距离”，也就是每轮可能的最大损失，这个损失累积了 $T$ 轮，所以 $GDT$ 就是 Regret 可能的最大值，是非常 trivial 的。

推导上界一定得是一个有意义的上界✍️✍️✍️。我们的 Regret 上界至少得是关于 $T$ 亚线性的，亚线性也意味着 $\lim _{T\to\infty}\frac{Reg _T}{T}\le 0$，从而学习者的平均累积损失渐进地不差于最优决策 $\mathbf{x} _\star$ 的平均累积损失。本文的 setting 允许我们得到一个 $\mathcal{O}(GD\sqrt{T})$ 上界，并证明（在某种意义上）无法做到更优。

本节的最后，也是在很多经典文献中证明 Regret 上界的第一步，就是做**线性化**：利用 $f _t(\mathbf{x})$ 为凸函数这一假设：<div id="regret-linearized"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
Reg _T = \sum _{t=1}^T f _t(\x _t) - \sum _{t=1}^T f _t(\xs) \le \sum _{t=1}^T \langle \nabla f _t(\x _t), \x _t - \xs \rangle. \tag{9}
$$

这一步线性化相当于将在线凸优化的凸函数限定在了线性函数，因此也被称为 Online Linear Optimization（OLO）。

你可能会说，太好了，我们第一步就建立在 $f _t(\mathbf{x})$ 为凸函数的假设上，那要是非凸，后面不就白做了？——倒也没有那么不堪。非凸意味着我们没法如此轻松地做线性化，但不意味着没法做——一旦我们通过某种方式在非凸优化问题中引入了类似式 [$(9)$](#regret-linearized) 的线性刻画（例如 [Online-to-Non-convex Conversion](https://arxiv.org/abs/2302.03775)），OLO 所做的研究就能立即为其所用。

## 2. Online Gradient Descent

### 2.1 Why Online Gradient Descent?

什么样的在线学习算法可以给 Regret 有意义的上界？或许这一节的标题有点剧透，但是我们还是从一些简单且基本的东西开始。当然，本文我们给出的算法只是若干可行路径中的一条，我们引出这个算法的思路也不会是它一开始被设计出来的思路，如果你觉得你有新的想法——非常推荐尝试分析它！

考虑 $\mathbf{x} _t$ 到 $\mathbf{x} _{t+1}$ 怎么更新——当然得是根据损失函数来设计，这里也就是式 [$(9)$](#regret-linearized)。先不管梯度，光是式 [$(9)$](#regret-linearized) 的内积形式以及 $\mathbf{x} _t,\mathbf{x} _{t+1},\mathbf{x} _\star$ 三个点，是否能让你想起一个基础的数学公式？我要说的是，**余弦定理**：
$$
\def \a {\mathbf{a}}
\def \b {\mathbf{b}}
\def \c {\mathbf{c}}
2\langle \a - \b, \a - \c \rangle = \Vert\a - \b\Vert^2 + \Vert\a - \c\Vert^2 - \Vert\b - \c\Vert^2.
$$
这里的 $\mathbf{a,b,c}$ 分别该选谁呢？反正，我已经帮你选完了：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
2\langle \x _t - \x _{t+1}, \x _t - \xs \rangle = \Vert\x _t - \x _{t+1}\Vert^2 + \Vert\x _t - \xs\Vert^2 - \Vert\x _{t+1} - \xs\Vert^2.
$$
注意，到目前为止，我们还没有指定任何 $\mathbf{x} _t$ 和 $\mathbf{x} _{t+1}$ 之间的关系，也就是适用于任意序列 $\\{\mathbf{x} _t\\} _{t=1}^T$。对 $t=1$ 到 $T$ 求和，你会发现它可以**错位相消**（telescoping）：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
2\sum _{t=1}^T \langle \x _t - \x _{t+1}, \x _t - \xs \rangle = \sum _{t=1}^T \Vert\x _t - \x _{t+1}\Vert^2 + \Vert\x _1 - \xs\Vert^2 - \Vert\x _{T+1} - \xs\Vert^2. \tag{10}
$$
能做到错位相消至少说明它还算是一个有点用的结构。现在让我们拿出线性化后的 Regret 对比一下：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
Reg _T = \sum _{t=1}^T f _t(\x _t) - \sum _{t=1}^T f _t(\xs) \le \sum _{t=1}^T \langle \nabla f _t(\x _t), \x _t - \xs \rangle,
$$
观察二者形式，我们发现最好让 $\mathbf{x} _t-\mathbf{x} _{t+1}$ 和 $\nabla f _t(\mathbf{x} _t)$ 对应——当然我们还可以引入步长 $\eta$ 作为算法的参数——也就是 $\mathbf{x} _t-\mathbf{x} _{t+1}=\eta\nabla f _t(\mathbf{x} _t)$，即 $\mathbf{x} _{t+1}=\mathbf{x} _t-\eta\nabla f _t(\mathbf{x} _t)$。因为 $\mathbf{x} _{t+1}$ 需要位于可行域 $\mathcal{X}$ 内，所以还要加一步投影操作。很好！我们得到了 OCO 问题最经典的算法框架之一：**在线梯度下降**（Online Gradient Descent, OGD），即<div id="ogd"></div>

$$
\def \x {\mathbf{x}}
\def \xt {\widetilde{\x}}
\xt _{t+1} = \x _t - \eta _t \nabla f _t(\x _t),\quad \x _{t+1} = \Pi _{\mathcal{X}}[ \xt _{t+1} ], \tag{11}
$$
其中 $\eta _t>0$ 是步长，$\Pi _{\mathcal{X}}[\cdot]$ 是向凸集 $\mathcal{X}$ 的投影操作。

### 2.2 Regret Analysis

现在让我们回到分析中。OGD 式 [$(11)$](#ogd) 代入式 [$(9)$](#regret-linearized) Regret 可得：<div id="gradient-descent-regret"></div>
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\def \xt {\widetilde{\x}}
\begin{align*}
Reg _T &\le \sum _{t=1}^T \langle \nabla f _t(\x _t), \x _t - \xs\rangle = \sum _{t=1}^T \frac{1}{\eta _t}\langle \x _t - \xt _{t+1}, \x _t - \xs\rangle \\\\
&= \sum _{t=1}^T \frac{1}{2\eta _t} \Big(\Vert \x _t - \xt _{t+1} \Vert^2 + \Vert\x _t - \xs\Vert^2 - \Vert\xt _{t+1} - \xs\Vert^2 \Big) \\\\
&\le \sum _{t=1}^T \frac{1}{2\eta _t} \Big(\Vert \x _t - \xt _{t+1} \Vert^2 + \Vert\x _t - \xs\Vert^2 - \Vert\x _{t+1} - \xs\Vert^2 \Big) \\\\
&= \frac{1}{2}\sum _{t=1}^T \eta _t\Vert \nabla f _t(\x _t) \Vert^2 + \sum _{t=1}^T \frac{1}{2\eta _t}\Big(\Vert\x _t - \xs\Vert^2 - \Vert\x _{t+1} - \xs\Vert^2 \Big). \tag{12}
\end{align*}
$$
其中第三行，我们使用了投影的性质（毕达哥拉斯定理）：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\def \y {\mathbf{y}}
\def \xt {\widetilde{\x}}
\x = \Pi _{\mathcal{X}}[ \xt ],\quad\implies\quad \forall \y\in\mathcal{X},\Vert\x-\y\Vert\le \Vert\xt - \y\Vert.
$$

式 [$(12)$](#gradient-descent-regret) 可以看出有两部分：第一部分梯度范数累积正比于步长，第二部分 telescoping 反比于步长。大致理解来看，我们可以认为第一部分是算法的 variance；第二部分是算法的 bias（这个含义会在下一篇文章的另一种分析中被显式地拆解出来，挖坑）。它也体现了一个在线学习算法永远在做的一件事：在探索与利用之间做平衡——每一步更新既要根据新信息做及时的更新，也要充分利用已有信息——平衡的关键就在于步长的设计。对于在线梯度下降来说，从上一步用梯度走多远，就是探索与利用的过程。这个思想在很多领域都有所体现。

——所以步长怎么设计呢？

### 2.3 Step-size Design

先考虑简单情况。如果 $\eta _t\equiv \eta$，则式 [$(12)$](#gradient-descent-regret)： 
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
Reg _T \le \frac{\eta}{2} \sum _{t=1}^T\Vert\nabla f _t(\x _t)\Vert^2 + \frac{1}{2\eta}\Vert\x _1 - \xs\Vert^2 \le \frac{\eta G^2 T}{2} + \frac{D^2}{2\eta} = GD\sqrt{T}. \tag{13}
$$
这是一个很明显的关于步长 $\eta$ 的 trade-off，解得 $\eta=\frac{D}{G\sqrt{T}}$。

或者我们保留变步长 $\eta _t$，考虑一个更有意思的设计。具体需要利用 Self-confident Tuning Lemma 这一数学工具，假设 $a  _1 ,\dots, a  _T$ 是非负实数，则：
$$
\sqrt{\sum _{t=1}^T a _t} \le \sum _{t=1}^T\frac{a _t}{\sqrt{\sum _{s=1}^t a _s}}\le 2\sqrt{\sum _{t=1}^T a _t}. \tag{14}
$$
现在参照该引理，我们定义变步长 $\eta _t=\frac{\alpha}{\sqrt{\sum _{s=1}^t\Vert\nabla f _s(\mathbf{x} _s)\Vert^2}}$（称为 “adaptive step-size”，$\alpha>0$ 为待定常数）。回到式 [$(12)$](#gradient-descent-regret)：<div id="gradient-descent-adaptive-bound"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\begin{align*}
Reg _T &\le \frac{1}{2}\sum _{t=1}^T \eta _t\Vert \nabla f _t(\x _t) \Vert^2 + \sum _{t=2}^T \left(\frac{1}{2\eta _t} - \frac{1}{2\eta _{t-1}}\right) \Vert\x _t - \xs\Vert^2 + \frac{1}{2\eta _1}\Vert\x _1 - \xs\Vert^2 \\\\
&\le \frac{1}{2}\sum _{t=1}^T \eta _t\Vert \nabla f _t(\x _t) \Vert^2 + \sum _{t=2}^T \left(\frac{1}{2\eta _t} - \frac{1}{2\eta _{t-1}}\right) D^2 + \frac{D^2}{2\eta _1} \\\\
&= \frac{1}{2}\sum _{t=1}^T \eta _t\Vert \nabla f _t(\x _t) \Vert^2 + \frac{D^2}{2\eta _T} \\\\
&= \frac{\alpha}{2}\sum _{t=1}^T \frac{\Vert\nabla f _t(\x _t)\Vert^2}{\sqrt{\sum _{s=1}^t\Vert\nabla f _s(\x _s)\Vert^2}} + \frac{D^2\sqrt{\sum _{t=1}^T\Vert\nabla f _t(\x _t)\Vert^2}}{2\alpha} \\\\
&\le \left(\alpha + \frac{D^2}{2\alpha}\right) \sqrt{\sum _{t=1}^T\Vert\nabla f _t(\x _t)\Vert^2} = \mathcal{O}\left( D\sqrt{\sum _{t=1}^T\Vert\nabla f _t(\x _t)\Vert^2}\right) = \mathcal{O}(GD\sqrt{T}), \tag{15}    
\end{align*}
$$
其中我们令 $\alpha=D$——这又是一个 trade-off，相较于之前的 $\eta=\frac{D}{G\sqrt{T}}$，**adaptive** step-size 做到了**自适应**梯度那部分，减轻了我们的 trade-off 压力。Adaptive step-size 还有一个好处是算法只需要事先知道 $D$——由于投影操作显式地需要可行域，这一般是已知的。不需要 $T$ 意味着算法可以运行任意轮数且在每一轮都有理论保障，这也称为 “anytime” 算法。

还记得我们一开始提到的 Adam 的祖宗——[AdaGrad](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36483.pdf) 吗？AdaGrad 就是将我们刚见到的 [adaptive learning rate](https://arxiv.org/abs/1002.4862)（adaptive step-size）直接迁移到了 entry-wise。你也可以从 Adam 的步长设计中，看到 $\frac{\alpha}{\sqrt{\sum _{s=1}^t\Vert\nabla f _s(\mathbf{x} _s)\Vert^2}}$ 的影子。

## 3. Lower Bound

回顾证明，我们主要在两个地方使用了不等号：第一处是线性化，第二处是利用投影性质做的放缩。这两个放缩会不会导致我们的上界 $\mathcal{O}(GD\sqrt{T})$ 太松了呢？本节将会说明，对于“**最坏情况**”，它是紧的。

由于我们证明的上界对于所有 OCO setting 都成立，为了说明没有更优的算法或者分析方法能够进一步从阶上改进这个上界，就需要证明对于任意算法，总存在一个最坏情况使得该算法具有 $\Omega(GD\sqrt{T})$ 的 regret 下界。也就是说，我们尝试证明：

$$
\def \x {\mathbf{x}}
\inf _{\text{algorithm}} \sup _{\text{environment}} Reg _T \ge \Omega(GD\sqrt{T}). \tag{16}
$$

本文采用的证明思路为，当确定可行域 $\mathcal{X}$ 和在线函数集 $\mathcal{F}$ 后，我们可以直接求解以 regret 为博弈指标的 minimax 值：
$$
\def \x {\mathbf{x}}
\def \X {\mathcal{X}}
\def \F {\mathcal{F}}
\mathcal{G} _T\triangleq \inf _{\x _1\in\X} \sup _{f _1\in\F} \inf _{\x _2\in\X} \sup _{f _2\in\F} \cdots \inf _{\x _T\in\X} \sup _{f _T\in\F} \left(\sum _{t=1}^T f _t(\x _t) - \min _{\x\in\X}\sum _{t=1}^T f _t(\x) \right). \tag{17}
$$

所以我们只需要构造出一组 $(\mathcal{X,F})$，并证明此时 $\mathcal{G} _T\ge \Omega(GD\sqrt{T})$ 即可。我们考虑一个简单（但是依然不容易）的 case：令 $\mathcal{X}=\\{\mathbf{x}:\Vert\mathbf{x}\Vert\le D\\}$，以及 $\mathcal{F}=\\{f(\mathbf{x})=\mathbf{g}^\top\mathbf{x}: \Vert\mathbf{g}\Vert\le G \\}$.

首先根据 $\mathcal{X}$，可以解出 $\mathbf{x} _\star$ 并得到：
$$
\def \x {\mathbf{x}}
\def \g {\mathbf{g}}
\def \X {\mathcal{X}}
\def \F {\mathcal{F}}
\mathcal{G} _T = \inf _{\Vert\x _1\Vert\le D} \sup _{\Vert\g _1\Vert\le G}\cdots \inf _{\Vert\x _T\Vert\le D} \sup _{\Vert\g _T\Vert\le G} \left(\sum _{t=1}^T \g _t^\top \x _t + D \Vert\g _{1:T}\Vert \right). \tag{18}
$$

根据该式推导博弈双方的策略。倒过来推，对于最后一手，环境只需考虑利用 $\mathbf{g} _T$ 最大化：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \langle\g _T,\mathbf{x} _T\rangle + D\Vert\g _T + \g _{1:T-1}\Vert\\\\
={} & \langle\g _T,\mathbf{x} _T\rangle + D\sqrt{\Vert\g _T\Vert^2+\Vert\g _{1:T-1}\Vert^2 + 2\langle \g _T ,\g _{1:T-1}\rangle}.  
\end{align*}
$$
显然对于环境而言，最好能找到一个与 $\mathbf{x} _T$ 和 $\mathbf{g} _{1:T-1}$ 夹脚都为正的方向，然后模长拉满到 $G$。据此倒推，学习者抵抗这一策略的方法就是让 $\mathbf{x} _T$ 恰好为 $\mathbf{g} _{1:T-1}$ 的反方向。令 $\mathbf{x} _T= -d _T \frac{\mathbf{g} _{1:T-1}}{\Vert\mathbf{g} _{1:T-1}\Vert}$，$d _T\ge 0$ 待定，此时再令环境通过 $\mathbf{g} _T$ 最大化：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \langle\g _T,\mathbf{x} _T\rangle + D\sqrt{\Vert\g _T\Vert^2+\Vert\g _{1:T-1}\Vert^2 + 2\langle \g _T ,\g _{1:T-1}\rangle} \\\\
={} & \frac{-d _T}{\Vert\g _{1:T-1}\Vert}\langle\g _T,\g _{1:T-1}\rangle + D\sqrt{\Vert\g _T\Vert^2+\Vert\g _{1:T-1}\Vert^2 + 2\langle \g _T ,\g _{1:T-1}\rangle} \\\\
={} & \frac{-d _T}{\Vert\g _{1:T-1}\Vert}a + D\sqrt{G^2+\Vert\g _{1:T-1}\Vert^2 + 2a}, \tag*{($a\triangleq \langle \g _T ,\g _{1:T-1}\rangle$)}
\end{align*}
$$
其中我们还令 $\Vert\mathbf{g} _T\Vert$ 先拉到最大模长 $G$，同时也不影响 $a$ 取值。上式对 $a$ 求导：
$$
\def \g {\mathbf{g}}
\begin{align*}
&\frac{-d _T}{\Vert\g _{1:T-1}\Vert} + \frac{D}{\sqrt{G^2+\Vert\g _{1:T-1}\Vert^2 + 2a}} = 0, \\\\
\implies{} & 2a = \frac{D^2}{d _T^2}\Vert\g _{1:T-1}\Vert^2 - G^2 - \Vert\g _{1:T-1}\Vert^2.
\end{align*}
$$
将 $a$ 代回，然后求解使最小化的 $d _T$：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \frac{-d _T}{\Vert\g _{1:T-1}\Vert}a + D\sqrt{G^2+\Vert\g _{1:T-1}\Vert^2 + 2a} \\\\
={} & \frac{D^2\Vert\g _{1:T-1}\Vert}{2d _T} + \frac{d _T(G^2 + \Vert\g _{1:T-1}\Vert^2)}{2\Vert\g _{1:T-1}\Vert} \\\\
={} & D\sqrt{G^2 + \Vert\g _{1:T-1}\Vert^2}. \tag*{$(d _T = D\frac{\Vert\g _{1:T-1}\Vert}{\sqrt{G^2 + \Vert\g _{1:T-1}\Vert^2}})$}
\end{align*}
$$
于是我们解出了学习者的最优策略。$d _T$ 代回 $a$ 可知 $a=0$，所以环境的最优策略是令 $\mathbf{g} _T$ 与 $\mathbf{x} _T$ 和 $\mathbf{g} _{1:T-1}$ 分别正交且模长拉满，这在维度大等于 $3$ 时始终可以做到。

通过数学归纳法，每一轮学习者和环境都采用该策略，于是
$$
\def \x {\mathbf{x}}
\def \g {\mathbf{g}}
\begin{align*}
\mathcal{G} _T &= \inf _{\Vert\x _1\Vert\le D} \sup _{\Vert\g _1\Vert\le G}\cdots \inf _{\Vert\x _T\Vert\le D} \sup _{\Vert\g _T\Vert\le G} \left(\sum _{t=1}^T \g _t^\top \x _t + D \Vert\g _{1:T}\Vert \right) \\\\
&= D\sqrt{G^2 + \Vert\g _{1:T-1}\Vert^2} = D\sqrt{G^2 + G^2 + \Vert\g _{1:T-2}\Vert^2} = \cdots \\\\
&= GD \sqrt{T}. \tag{19}
\end{align*}
$$

这就是该 $(\mathcal{X,F})$ setting 下对于最优算法能得到的 regret，任何算法的 regret 上界都不可能小于这个值。因此，我们之前得到的 $GD\sqrt{T}$ 已经不可能从阶上、甚至是常数上有所改进了。

关于 $\Omega(GD\sqrt{T})$ 的证明还有一种令在线函数随机化的 setting，虽然得到的下界比本文在常数上小一些，但是或许更加涉及本质。感兴趣的读者可以阅读[这个博客](https://parameterfree.com/2019/09/25/lower-bounds-for-online-linear-optimization/)的第一节。

## 4. Worst-case / Adaptive Bounds

在上文中，我们已经基本了解在最经典的凸函数 setting 下使用在线梯度下降算法优化的 regret 上界，并证明从阶上无法再改进。这种上界也被称为 “worst-case bound”，也就是环境从始至终都和学习者对着干。

当然，还有很多情况下环境并没有这么具有对抗性，这种情况下有些算法就能够自适应地改进对应的 regret 上界，得到“adaptive bound”，它只有当环境变得完全对抗时才会退化成 worst-case 对应的情况。一个简单的例子就是式 [$(9)$](#gradient-descent-adaptive-bound) 的最后一行，我们已经得到一个 $\mathcal{O}(D\sqrt{\sum _{t=1}^T\Vert\nabla f _t(\mathbf{x} _t)\Vert^2})$ 的 adaptive bound。只有当每一轮的梯度都拉满到 $G$ 时，才会退化回 $\mathcal{O}(GD\sqrt{T})$ 的 worst-case bound；而其他情况下它可以变得更小。

在下一篇中，我们会改进在线梯度下降算法，让它更具适应性！我们还会发现，这类算法可以和很多领域产生关联，比如优化领域的加速方法、零和博弈里的加速求解纳什均衡，等等。