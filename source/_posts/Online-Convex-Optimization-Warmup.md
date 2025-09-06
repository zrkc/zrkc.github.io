---
title: Online Convex Optimization Warmup
date: 2025-09-06 11:01:07
tags:
---

> 本文是为高级优化课程的学生在正式学习在线凸优化之前准备的。具体来说，本文介绍了在线凸优化的一个典型 setting，给出经典算法，并推导了上下界。  
> 本文的目的是希望读者能在正式学习在线凸优化之前，先对“在线学习”有个大致的认识，以免陷于数学证明之中失去了最宝贵的兴趣。为照顾逻辑表达，本文行文叙述和理论推导可能不会十分严谨，希望读者注意，也欢迎给出意见！

## 1. Problem Formulation

在线凸优化是在线学习（Online Learning）的一个 case。在线学习简单来说就是设计算法使得能够随着环境的变化不断更新自身模型适应它，例如：与环境产生交互、数据按批次处理等问题。

**【这里可能需要一个引子，比如随机优化和 Adam】**


我们将它抽象成学习者与环境的博弈，即如下范式：

**在线学习.**  
已知决策可行域 $\mathcal{X}\subseteq\mathbb{R}^d$。在第 $t=1,2,\dots$ 轮：
1. 学习者选择决策 $\mathbf{x}_t\in\mathcal{X}\subseteq \mathbb{R}^d$；
2. 环境给出在线函数 $f_t:\mathcal{X}\to\mathbb{R}$；
3. 学习者受到 $f_t(\mathbf{x}_t)$ 的损失，观测到某些 $f_t$ 的信息并更新决策。

在线凸优化（Online Convex Optimization, OCO）研究可行域 $\mathcal{X}$ 为凸集且在线函数 $f_t$ 均为凸函数的情况。

### 1.1 Performance Measure

最直接地，我们希望最小化累积损失
$$
\def \x {\mathbf{x}}
\sum_{t=1}^T f_t(\x_t). 
$$

但这个指标很不公平、导致难以下手，因为环境可以对抗性地根据 $\mathbf{x}_t$ 给出 $f_t$。类似超额风险（模型损失与最优模型损失之差）引入最优决策作为 comparator 限制优化目标的难度，我们采用称为“遗憾”（regret）的度量指标，定义为：

<div id="regret"></div>

$$
\def \x {\mathbf{x}}
Reg_T \triangleq \sum_{t=1}^T f_t(\x_t) - \min_{\x\in\mathcal{X}}\sum_{t=1}^T f_t(\x). \tag{1}
$$

我们简记 $\mathbf{x}_\star\triangleq \argmin_{\mathbf{x}\in\mathcal{X}}\sum_{t=1}^T f_t(\mathbf{x}_t)$。学习者的目标是最小化遗憾，由于环境的决策此时也会影响 $\mathbf{x}_\star$ 的累积损失，所以这是一个合理的博弈。

在本文中，我们还同时关注两个 OCO 中比较常见的假设：可行域有界假设，和梯度有界假设。

- **可行域有界假设.** 对任意 $\mathbf{x,y}\in\mathcal{X}$，有 $\|\mathbf{x-y}\|_2\le D$.  
- **梯度有界假设.** 对任意 $\mathbf{x}\in\mathcal{X}$ 和 $t\in[T]$，有 $\|\nabla f_t(\mathbf{x})\|_2\le G$。即在线函数是 $G$-Lipschitz 的。

下文简记 $\ell_2$-范数 $\|\cdot\|_2$ 为 $\|\cdot\|$。引入这两个假设最直接的作用就是简化了算法设计和证明。这两个假设也很基本，从量纲上来说分别对应了距离和斜率，乘起来就是损失。当然也有很多工作在研究不依赖这两个假设或者对应参数的算法，至今仍有很多有待研究的空间。

### 1.2 A Trivial Example

下文中，我们要做的就是设计算法，并证明 regret 有上界保证——通常用渐进复杂度上界表示——并且至少得是关于 $T$ 亚线性的。为什么会这么考虑呢？首先，亚线性意味着平均意义下相对于最优决策的损失之差会趋于零：$\lim_{T\to\infty}\frac{Reg_T}{T}=0$，从而累积损失趋近于全局最优决策 $\mathbf{x}_\star$ 的累计损失。其次，一个关于 $T$ 线性的上界是 trivial 的。最简单地，根据微分中值定理：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\def \c {\mathbf{c}}
\begin{align*}
Reg_T &= \sum_{t=1}^T f_t(\x_t) - \sum_{t=1}^T f_t(\xs) = \sum_{t=1}^T \langle \nabla f_t(\c_t), \x_t - \xs \rangle \qquad(\c_t\in[\x_t,\xs]) \\
&\le \sum_{t=1}^T \| \nabla f_t(\c_t) \| \|\x_t - \xs\| \le GDT.
\end{align*}
$$
其中第一个不等号使用了 Hölder 不等式。我们可以稍微观察一下这个上界：$GD$ 是最大“斜率”乘最大“距离”，也就是每轮可能的最大损失，这个损失累积了 $T$ 轮，所以 $GDT$ 就是 regret 可能的最大值，是非常 trivial 的。

推导上界一定得是一个有意义的上界✍️。在后文中，我们会最终得到一个亚线性的 $\mathcal{O}(GD\sqrt{T})$ 上界，并证明（在某种意义上）无法做到更优。

本节的最后，也是在很多经典文献中证明 regret 上界的第一步，就是做线性化：即利用 $f_t(\mathbf{x})$ 为凸函数这一假设：

<div id="regret-linearized"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
Reg_T = \sum_{t=1}^T f_t(\x_t) - \sum_{t=1}^T f_t(\xs) \le \sum_{t=1}^T \langle \nabla f_t(\x_t), \x_t - \xs \rangle. \tag{2}
$$

相比刚才的微分中值定理，好处是我们不需要知道那个中值点 $\mathbf{c}_t$，坏处也很显然，假设函数为凸本身就局限了它的应用。无论如何，接下来可以设计我们的第一个在线凸优化算法了！

## 2. Online Gradient Descent

### 2.1 Why Online Gradient Descent?

OCO 问题最经典的算法框架之一是在线梯度下降（Online Gradient Descent, OGD），即
$$
\def \x {\mathbf{x}}
\def \xt {\widetilde{\x}}
\xt_{t+1} = \x_t - \eta_t \nabla f_t(\x_t),\quad \x_{t+1} = \Pi_{\mathcal{X}}[ \xt_{t+1} ],
$$
其中 $\eta_t>0$ 是步长，$\Pi_{\mathcal{X}}[\cdot]$ 是向凸集 $\mathcal{X}$ 的投影操作。可能你立即会问了，为什么：一定得从上一步做更新而不是上上步甚至随便一个点？从博弈过程来看，$f_t$ 和 $f_{t+1}$ 似乎没什么联系，那么从 $\mathbf{x}_t$ 拿着 $f_t$ 在该点的梯度去更新得到 $\mathbf{x}_{t+1}$，对于优化 $f_{t+1}$ 能有多大帮助呢？

这里主要解释直观上为什么 OGD 能 work。我们先抛开 OCO 的语义，从数学角度看这个问题。在后文中我们会经常见到一类数学结构，它们看起来非常类似余弦定理：
$$
\def \a {\mathbf{a}}
\def \b {\mathbf{b}}
\def \c {\mathbf{c}}
2\langle \a - \b, \a - \c \rangle = \|\a - \b\|^2 + \|\a - \c\|^2 - \|\b - \c\|^2.
$$

我们观察式 [$(2)$](#regret-linearized) 线性化后的 regret：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\sum_{t=1}^T \langle \nabla f_t(\x_t), \x_t - \xs\rangle,
$$

同样是内积式。如果需要将 regret 和余弦定理联系起来，一个思路是观察到向余弦定理代入：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
2\langle \x_t - \x_{t+1}, \x_t - \xs \rangle = \|\x_t - \x_{t+1}\|^2 + \|\x_t - \xs\|^2 - \|\x_{t+1} - \xs\|^2.
$$

这个内积式目前还不依赖任何 $\mathbf{x}_t$ 和 $\mathbf{x}_{t+1}$ 之间的关系，也就是适用于任意序列 $\{\mathbf{x}_t\}_{t=1}^T$。对 $t=1$ 到 $T$ 求和，会发现它可以**错位相消**（telescoping）：

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
2\sum_{t=1}^T \langle \x_t - \x_{t+1}, \x_t - \xs \rangle = \sum_{t=1}^T \|\x_t - \x_{t+1}\|^2 + \|\x_1 - \xs\|^2 - \|\x_{T+1} - \xs\|^2.
$$

能做到错位相消是一个我们比较满意的结构。为了让它和 regret 沾上边，对比二者形式我们发现最好让 $\mathbf{x}_t-\mathbf{x}_{t+1}$ 和 $\nabla f_t(\mathbf{x}_t)$ 对应。当然我们还可以引入步长 $\eta$ 作为算法的参数。如果不考虑投影操作，最简单的设计就是 $\mathbf{x}_t-\mathbf{x}_{t+1}=\eta\nabla f_t(\mathbf{x}_t)$，它已经很类似在线梯度下降的更新式了。代入 regret 可得：<div id="regret-telescoping"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\begin{align*}
Reg_T &\le \sum_{t=1}^T \langle \nabla f_t(\x_t), \x_t - \xs \rangle = \frac{1}{\eta}\sum_{t=1}^T \langle \x_t - \x_{t+1}, \x_t - \xs \rangle \\
&= \frac{1}{2\eta}\left(\sum_{t=1}^T \|\x_t - \x_{t+1}\|^2 + \|\x_1 - \xs\|^2 - \|\x_{T+1} - \xs\|^2. \right) \\
&= \frac{\eta}{2}\sum_{t=1}^T \|\nabla f_t(\x_t)\|^2 + \frac{1}{2\eta}\left(\|\x_1 - \xs\|^2 - \|\x_{T+1} - \xs\|^2\right). \tag{3}   
\end{align*}
$$

令人高兴的是，我们可以进一步利用 $\eta$ 在上式做 trade-off！可以大概预想到，通过合理设置 $\eta$，我们能得到一个亚线性，具体而言，$\mathcal{O}(\sqrt{T})$ 的上界。

在阅读下一小节之前，你可能会有一些自己的想法。你可以试试自己推导看看，这会很快帮助你形成一个比较合理的认知 :)

### 2.2 Some Questions

到这里，不知道你会不会问：为什么不做“梯度上升”呢？从分析上看似乎也有些道理：上面的余弦定理是两个正项和一个负项，梯度上升意味着使用余弦定理取负，变成两个负项一个正项，而且依然保留 telescoping 的结构！但是，不能这么做的原因在于，那个原本为负、现在为正的项，最终会变得难以控制地大。

例如，令 $\mathbf{x}_t-\mathbf{x}_{t+1}=-\eta\nabla f_t(\mathbf{x}_t)$，代入：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\eta\langle \nabla f_t(\x_t), \x_t - \xs\rangle = -
\langle \x_t - \x_{t+1}, \x_t - \xs \rangle = \frac{1}{2}\Big(\|\x_{t+1} - \xs\|^2 - \|\x_t - \xs\|^2 - \|\x_t - \x_{t+1}\|^2\Big).
$$

对 $t=1$ 到 $T$ 求和（注意我们忽略了投影）：<div id="gradient-ascent"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
Reg_T \le \frac{1}{2\eta}\left(\|\x_{T+1} - \xs\|^2 - \|\x_1 - \xs\|^2 - \sum_{t=1}^T \|\x_t - \x_{t+1}\|^2\right) . \tag{4}
$$

看起来只有唯一的正项 $\|\mathbf{x}_{T+1}-\mathbf{x}_\star\|^2$，莫非这是一个好事？但这还没完。式 [$(3)$](#regret-telescoping) 中的正项 $\|\mathbf{x}_1 - \mathbf{x}_\star\|^2$，通常来说，我们会认为它是一个和初始点选择有关的常数；而式 [$(4)$](#gradient-ascent) 中的正项 $\|\mathbf{x}_{T+1}-\mathbf{x}_\star\|^2$ 就是一个和算法运行过程相关的量了。因为推导忽略了投影，相当于令 $\mathcal{X}=\mathbb{R}^d$，所以此时可行域有界假设不成立；万一 $\|\mathbf{x}_{T+1}-\mathbf{x}_\star\|^2$ 随着 $T$ 增长速度是线性的呢？这也体现了证明上界的一个基本原则：虽然证明过程中产生了很多大等于 regret 的式子，但是只有当这个式子有普适意义时（例如，和算法运行过程量无关，体现为某些全局参数的函数），才能够和其他算法对比。下一步，我们顶多利用 Cauchy–Schwarz 不等式：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\begin{align*}
& \|\x_{T+1} - \xs\|^2 = \|\x_{T+1}-\x_T+\x_T - \x_{T-1} + \cdots + \x_2 - \x_1 + \x_1 - \xs\|^2 \\
\le{} & (T+1)\left(\|\x_{T+1}-\x_T\|^2 + \|\x_T - \x_{T-1}\|^2 + \cdots + \|\x_2 - \x_1\|^2 + \|\x_1 - \xs\|^2 \right),
\end{align*}
$$

代入式 [$(4)$](#gradient-ascent) 得到：<div id="gradient-ascent-regret"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\begin{align*}
Reg_T &\le \frac{T}{2\eta}\left( \|\x_1 - \xs\|^2 + \sum_{t=1}^T \|\x_t - \x_{t+1}\|^2 \right) = \frac{T}{2}\left( \frac{1}{\eta}\|\x_1 - \xs\|^2 + \eta\sum_{t=1}^T \|\nabla f_t(\x_t)\|^2 \right). \tag{5}
\end{align*}
$$

很显然，和梯度下降的式 [$(3)$](#regret-telescoping) 相比，式 [$(5)$](#gradient-ascent-regret) 中 $\eta$ 最多只能在括号里头做 trade-off，而括号外还有一个线性的 $T$，高下立判！

总而言之，在线梯度下降能够 work 就是因为它合理地利用了余弦定理形式的 telescoping 结构。你可能还会问：为什么得从上一步而不是上上步或者其他呢？从上面的推导可以看出，从上一步只是为了方便做 telescoping 的衔接。事实上，不一定得从上一步开始，只要你能构造出一条（或几条）的链条就行了，比如：
$$
\def \x {\mathbf{x}}
\x_1 \underset{-\nabla f_1(\x_1)}{\longrightarrow} \x_3 \underset{-\nabla f_3(\x_3)}{\longrightarrow} \cdots,\quad \x_2 \underset{-\nabla f_2(\x_2)}{\longrightarrow} \x_4 \underset{-\nabla f_4(\x_4)}{\longrightarrow} \cdots.
$$
更夸张地，你还可以：
$$
\def \x {\mathbf{x}}
\x_1 \underset{-\nabla f_1(\x_1)}{\longrightarrow} \x_T ,\quad \x_2 \underset{-\nabla f_2(\x_2)}{\longrightarrow} \x_3 \underset{-\nabla f_3(\x_3)}{\longrightarrow} \cdots \underset{-\nabla f_{T-2}(\x_{T-2})}{\longrightarrow} \x_{T-1}.
$$

不过其实就相当于把在线函数分为两组，跑两个 OGD 算法，本质上还是脱离不开“从上一步更新”的框架。能做成 telescoping 的关键还在于要从 $\mathbf{x}_t$ 用这点的梯度信息 $\nabla f_t(\mathbf{x}_t)$ 更新，剩下的就是一些变体了。

下一节中，我们对在线梯度下降进行正式的理论推导，并得到第一个亚线性的上界。

## 3. Regret Analysis

### 3.1 Bias-Variance

回到在线梯度下降：<div id="gradient-descent"></div>

$$
\def \x {\mathbf{x}}
\def \xt {\widetilde{\x}}
\xt_{t+1} = \x_t - \eta_t \nabla f_t(\x_t),\quad \x_{t+1} = \Pi_{\mathcal{X}}[ \xt_{t+1} ]. \tag{6}
$$

虽然看起来多了一步投影，但是和上面的推导基本一样：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\def \xt {\widetilde{\x}}
\begin{align*}
Reg_T &\le \sum_{t=1}^T \langle \nabla f_t(\x_t), \x_t - \xs\rangle = \sum_{t=1}^T \frac{1}{\eta_t}\langle \x_t - \xt_{t+1}, \x_t - \xs\rangle \\
&= \sum_{t=1}^T \frac{1}{2\eta_t} \Big(\| \x_t - \xt_{t+1} \|^2 + \|\x_t - \xs\|^2 - \|\xt_{t+1} - \xs\|^2 \Big).
\end{align*}
$$

为了 telescoping，我们利用投影的性质（毕达哥拉斯定理）做一步放缩：
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\def \y {\mathbf{y}}
\def \xt {\widetilde{\x}}
\x = \Pi_{\mathcal{X}}[ \xt ],\quad\implies\quad \forall \y\in\mathcal{X},\|\x-\y\|\le \|\xt - \y\|.
$$

于是上式：<div id="gradient-descent-regret"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\def \xt {\widetilde{\x}}
\begin{align*}
Reg_T &\le \sum_{t=1}^T \frac{1}{2\eta_t} \Big(\| \x_t - \xt_{t+1} \|^2 + \|\x_t - \xs\|^2 - \|\x_{t+1} - \xs\|^2 \Big) \\
&= \frac{1}{2}\sum_{t=1}^T \eta_t\| \nabla f_t(\x_t) \|^2 + \sum_{t=1}^T \frac{1}{2\eta_t}\Big(\|\x_t - \xs\|^2 - \|\x_{t+1} - \xs\|^2 \Big). \tag{7}
\end{align*}
$$

该式子可以看出有两部分：第一部分梯度范数累积正比于步长，第二部分 telescoping 反比于步长。大致理解来看，我们可以认为第一部分是算法的 variance；第二部分是算法的 bias（这个含义在下一篇文章中更加明确地拆解出来，挖坑）。它也体现了一个在线学习算法永远在做的一件事：在探索与利用之间做平衡——每一步更新既要根据新信息做及时的更新，也要充分利用已有信息——平衡的关键就在于步长的设计。对于在线梯度下降来说，从上一步用梯度走多远，就是探索与利用的过程。这个思想在很多领域都有所体现。

在下一篇文章中，我们也会尝试更加显式地拆分出 trade-off 的两部分。这个拆分也将引出一类更加具有自适应能力的在线学习算法，并和更广泛的领域产生联系。

总之，我们先尝试设计步长以 trade-off。

### 3.2 Step-size Design

先考虑简单情况。如果 $\eta_t\equiv \eta$，则式 [$(7)$](#gradient-descent-regret)： 
$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
Reg_T \le \frac{\eta}{2} \sum_{t=1}^T\|\nabla f_t(\x_t)\|^2 + \frac{1}{2\eta}\|\x_1 - \xs\|^2 \le \frac{\eta G^2 T}{2} + \frac{D^2}{2\eta} = GD\sqrt{T}. \tag{8}
$$
其中我们令 $\eta=\frac{D}{G\sqrt{T}}$。

这里还可以考虑变步长 $\eta_t$，得到一个更有意思的设计。具体需要利用 Self-confident Tuning Lemma 这一数学工具，假设 $a _1 ,\dots, a _T$ 是非负实数，则：
$$
\sqrt{\sum_{t=1}^T a_t} \le \sum_{t=1}^T\frac{a_t}{\sqrt{\sum_{s=1}^t a_s}}\le 2\sqrt{\sum_{t=1}^T a_t}.
$$
现在参照上述引理，我们使用变步长并定义为 $\eta_t=\frac{\alpha}{\sqrt{\sum_{s=1}^t\|\nabla f_s(\mathbf{x}_s)\|^2}}$（其也称为 “adaptive step-size”，$\alpha>0$ 为待定常数）。回到式 [$(7)$](#gradient-descent-regret)：<div id="gradient-descent-adaptive-bound"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x_\star}
\begin{align*}
Reg_T &\le \frac{1}{2}\sum_{t=1}^T \eta_t\| \nabla f_t(\x_t) \|^2 + \sum_{t=2}^T \left(\frac{1}{2\eta_t} - \frac{1}{2\eta_{t-1}}\right) \|\x_t - \xs\|^2 + \frac{1}{2\eta_1}\|\x_1 - \xs\|^2 \\
&\le \frac{1}{2}\sum_{t=1}^T \eta_t\| \nabla f_t(\x_t) \|^2 + \frac{D^2}{2\eta_T} = \frac{\alpha}{2}\sum_{t=1}^T \frac{\|\nabla f_t(\x_t)\|^2}{\sqrt{\sum_{s=1}^t\|\nabla f_s(\x_s)\|^2}} + \frac{D^2\sqrt{\sum_{t=1}^T\|\nabla f_t(\x_t)\|^2}}{2\alpha} \\
&\le \left(\alpha + \frac{D^2}{2\alpha}\right) \sqrt{\sum_{t=1}^T\|\nabla f_t(\x_t)\|^2} = \mathcal{O}\left( D\sqrt{\sum_{t=1}^T\|\nabla f_t(\x_t)\|^2}\right) = \mathcal{O}(GD\sqrt{T}), \tag{9}    
\end{align*}
$$
其中我们令 $\alpha=D$。相较于 $\eta=\frac{D}{G\sqrt{T}}$，这里变步长的好处是算法只需要事先知道 $D$——由于投影操作显式地需要可行域，这一般是已知的。不需要 $T$ 意味着算法可以运行任意轮数，这也称为 “anytime” 算法。

## 4. Lower Bound

回顾证明，我们主要在两个地方使用了不等号：第一处是线性化，第二处是利用投影性质做的放缩。这两个放缩会不会导致我们的上界 $\mathcal{O}(GD\sqrt{T})$ 太松了呢？本节将会说明，对于“**最坏情况**”，它是紧的。

由于我们证明的上界对于所有 OCO setting 都成立，为了说明没有更优的算法或者分析方法能够进一步从阶上改进这个上界，就需要证明对于任意算法，总存在一个最坏情况使得该算法具有 $\Omega(GD\sqrt{T})$ 的 regret 下界。也就是说，我们尝试证明：

$$
\def \x {\mathbf{x}}
\inf_{\text{algorithm}} \sup_{\text{environment}} Reg_T \ge \Omega(GD\sqrt{T}). \tag{10}
$$

本文采用的证明思路为，当确定可行域 $\mathcal{X}$ 和在线函数集 $\mathcal{F}$ 后，我们可以直接求解以 regret 为博弈指标的 minimax 值：
$$
\def \x {\mathbf{x}}
\def \X {\mathcal{X}}
\def \F {\mathcal{F}}
\mathcal{G}_T\triangleq \inf_{\x_1\in\X} \sup_{f_1\in\F} \inf_{\x_2\in\X} \sup_{f_2\in\F} \cdots \inf_{\x_T\in\X} \sup_{f_T\in\F} \left(\sum_{t=1}^T f_t(\x_t) - \min_{\x\in\X}\sum_{t=1}^T f_t(\x) \right). \tag{11}
$$

所以我们只需要构造出一组 $(\mathcal{X,F})$，并证明此时 $\mathcal{G}_T\ge \Omega(GD\sqrt{T})$ 即可。我们考虑一个简单（但是依然不容易）的 case：令 $\mathcal{X}=\{\mathbf{x}:\|\mathbf{x}\|\le D\}$，以及 $\mathcal{F}=\{f(\mathbf{x})=\mathbf{g}^\top\mathbf{x}: \|\mathbf{g}\|\le G \}$.

首先根据 $\mathcal{X}$，可以解出 $\mathbf{x}_\star$ 并得到：
$$
\def \x {\mathbf{x}}
\def \g {\mathbf{g}}
\def \X {\mathcal{X}}
\def \F {\mathcal{F}}
\mathcal{G}_T = \inf_{\|\x_1\|\le D} \sup_{\|\g_1\|\le G}\cdots \inf_{\|\x_T\|\le D} \sup_{\|\g_T\|\le G} \left(\sum_{t=1}^T \g_t^\top \x_t + D \|\g_{1:T}\| \right). \tag{12}
$$

根据该式推导博弈双方的策略。倒过来推，对于最后一手，环境只需考虑利用 $\mathbf{g}_T$ 最大化：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \langle\g_T,\mathbf{x}_T\rangle + D\|\g_T + \g_{1:T-1}\|\\
={} & \langle\g_T,\mathbf{x}_T\rangle + D\sqrt{\|\g_T\|^2+\|\g_{1:T-1}\|^2 + 2\langle \g_T ,\g_{1:T-1}\rangle}.  
\end{align*}
$$
显然对于环境而言，最好能找到一个与 $\mathbf{x}_T$ 和 $\mathbf{g}_{1:T-1}$ 夹脚都为正的方向，然后模长拉满到 $G$。据此倒推，学习者抵抗这一策略的方法就是让 $\mathbf{x}_T$ 恰好为 $\mathbf{g}_{1:T-1}$ 的反方向。令 $\mathbf{x}_T= -d_T \frac{\mathbf{g}_{1:T-1}}{\|\mathbf{g}_{1:T-1}\|}$，$d_T\ge 0$ 待定，此时再令环境通过 $\mathbf{g}_T$ 最大化：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \langle\g_T,\mathbf{x}_T\rangle + D\sqrt{\|\g_T\|^2+\|\g_{1:T-1}\|^2 + 2\langle \g_T ,\g_{1:T-1}\rangle} \\
={} & \frac{-d_T}{\|\g_{1:T-1}\|}\langle\g_T,\g_{1:T-1}\rangle + D\sqrt{\|\g_T\|^2+\|\g_{1:T-1}\|^2 + 2\langle \g_T ,\g_{1:T-1}\rangle} \\
={} & \frac{-d_T}{\|\g_{1:T-1}\|}a + D\sqrt{G^2+\|\g_{1:T-1}\|^2 + 2a}, \qquad(a\triangleq \langle \g_T ,\g_{1:T-1}\rangle)
\end{align*}
$$
其中我们还令 $\|\mathbf{g}_T\|$ 先拉到最大模长 $G$，同时也不影响 $a$ 取值。上式对 $a$ 求导：
$$
\def \g {\mathbf{g}}
\begin{align*}
&\frac{-d_T}{\|\g_{1:T-1}\|} + \frac{D}{\sqrt{G^2+\|\g_{1:T-1}\|^2 + 2a}} = 0, \\
\implies{} & 2a = \frac{D^2}{d_T^2}\|\g_{1:T-1}\|^2 - G^2 - \|\g_{1:T-1}\|^2.
\end{align*}
$$
将 $a$ 代回，然后求解使最小化的 $d_T$：
$$
\def \g {\mathbf{g}}
\begin{align*}
& \frac{-d_T}{\|\g_{1:T-1}\|}a + D\sqrt{G^2+\|\g_{1:T-1}\|^2 + 2a} \\
={} & \frac{D^2\|\g_{1:T-1}\|}{2d_T} + \frac{d_T(G^2 + \|\g_{1:T-1}\|^2)}{2\|\g_{1:T-1}\|} \\
={} & D\sqrt{G^2 + \|\g_{1:T-1}\|^2}. \tag*{$(d_T = D\frac{\|\g_{1:T-1}\|}{\sqrt{G^2 + \|\g_{1:T-1}\|^2}})$}
\end{align*}
$$
于是我们解出了学习者的最优策略。$d_T$ 代回 $a$ 可知 $a=0$，所以环境的最优策略是令 $\mathbf{g}_T$ 与 $\mathbf{x}_T$ 和 $\mathbf{g}_{1:T-1}$ 分别正交且模长拉满，这在维度大等于 $3$ 时始终可以做到。

通过数学归纳法，每一轮学习者和环境都采用该策略，于是
$$
\def \x {\mathbf{x}}
\def \g {\mathbf{g}}
\begin{align*}
\mathcal{G}_T &= \inf_{\|\x_1\|\le D} \sup_{\|\g_1\|\le G}\cdots \inf_{\|\x_T\|\le D} \sup_{\|\g_T\|\le G} \left(\sum_{t=1}^T \g_t^\top \x_t + D \|\g_{1:T}\| \right) \\
&= D\sqrt{G^2 + \|\g_{1:T-1}\|^2} = D\sqrt{G^2 + G^2 + \|\g_{1:T-2}\|^2} = \cdots \\
&= GD \sqrt{T}. \tag{13}
\end{align*}
$$

这就是该 $(\mathcal{X,F})$ setting 下对于最优算法能得到的 regret，任何算法的 regret 上界都不可能小于这个值。因此，我们之前得到的 $GD\sqrt{T}$ 已经不可能从阶上、甚至是常数上有所改进了。

关于 $\Omega(GD\sqrt{T})$ 的证明还有一种令在线函数随机化的 setting，虽然得到的下界比本文在常数上小一些，但是或许更加涉及本质。感兴趣的读者可以阅读[这个博客](https://parameterfree.com/2019/09/25/lower-bounds-for-online-linear-optimization/)的第一节。

## 5. Worst-case / Adaptive Bounds

在上文中，我们已经基本了解在最经典的凸函数 setting 下使用在线梯度下降算法优化的 regret 上界，并证明从阶上无法再改进。这种上界也被称为 “worst-case bound”，也就是环境从始至终都和学习者对着干。

当然，还有很多情况下环境并没有这么具有对抗性，这种情况下有些算法就能够自适应地改进对应的 regret 上界，得到“adaptive bound”，它只有当环境变得完全对抗时才会退化成 worst-case 对应的情况。一个简单的例子就是式 [$(9)$](#gradient-descent-adaptive-bound) 的最后一行，我们已经得到一个 $\mathcal{O}(D\sqrt{\sum_{t=1}^T\|\nabla f_t(\mathbf{x}_t)\|^2})$ 的 adaptive bound。只有当每一轮的梯度都拉满到 $G$ 时，才会退化回 $\mathcal{O}(GD\sqrt{T})$ 的 worst-case bound；而其他情况下它可以变得更小。

在下一篇中，我们会改进在线梯度下降算法，让它更具适应性！我们还会发现，这类算法可以和很多领域产生关联，比如优化领域的加速方法、零和博弈里的加速求解纳什均衡，等等。