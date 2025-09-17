到这里，不知道你会不会问：为什么不做“梯度上升”呢？从分析上看似乎也有些道理：上面的余弦定理是两个正项和一个负项，梯度上升意味着使用余弦定理取负，变成两个负项一个正项，而且依然保留 telescoping 的结构！但是，不能这么做的原因在于，那个原本为负、现在为正的项，最终会变得难以控制地大。

例如，令 $\mathbf{x} _t-\mathbf{x} _{t+1}=-\eta\nabla f _t(\mathbf{x} _t)$，代入：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\eta\langle \nabla f _t(\x _t), \x _t - \xs\rangle = -
\langle \x _t - \x _{t+1}, \x _t - \xs \rangle = \frac{1}{2}\Big(\Vert\x _{t+1} - \xs\Vert^2 - \Vert\x _t - \xs\Vert^2 - \Vert\x _t - \x _{t+1}\Vert^2\Big).
$$

对 $t=1$ 到 $T$ 求和（注意我们忽略了投影）：<div id="gradient-ascent"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
Reg _T \le \frac{1}{2\eta}\left(\Vert\x _{T+1} - \xs\Vert^2 - \Vert\x _1 - \xs\Vert^2 - \sum _{t=1}^T \Vert\x _t - \x _{t+1}\Vert^2\right) . \tag{4}
$$

看起来只有唯一的正项 $\Vert\mathbf{x} _{T+1}-\mathbf{x} _\star\Vert^2$，莫非这是一个好事？但这还没完。式 [$(3)$](#regret-telescoping) 中的正项 $\Vert\mathbf{x} _1 - \mathbf{x} _\star\Vert^2$，通常来说，我们会认为它是一个和初始点选择有关的常数；而式 [$(4)$](#gradient-ascent) 中的正项 $\Vert\mathbf{x} _{T+1}-\mathbf{x} _\star\Vert^2$ 就是一个和算法运行过程相关的量了。因为推导忽略了投影，相当于令 $\mathcal{X}=\mathbb{R}^d$，所以此时可行域有界假设不成立；万一 $\Vert\mathbf{x} _{T+1}-\mathbf{x} _\star\Vert^2$ 随着 $T$ 增长速度是线性的呢？这也体现了证明上界的一个基本原则：虽然证明过程中产生了很多大等于 regret 的式子，但是只有当这个式子有普适意义时（例如，和算法运行过程量无关，体现为某些全局参数的函数），才能够和其他算法对比。下一步，我们顶多利用 Cauchy–Schwarz 不等式：
$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\begin{align*}
& \Vert\x _{T+1} - \xs\Vert^2 = \Vert\x _{T+1}-\x _T+\x _T - \x _{T-1} + \cdots + \x _2 - \x _1 + \x _1 - \xs\Vert^2 \\
\le{} & (T+1)\left(\Vert\x _{T+1}-\x _T\Vert^2 + \Vert\x _T - \x _{T-1}\Vert^2 + \cdots + \Vert\x _2 - \x _1\Vert^2 + \Vert\x _1 - \xs\Vert^2 \right),
\end{align*}
$$

代入式 [$(4)$](#gradient-ascent) 得到：<div id="gradient-ascent-regret"></div>

$$
\def \x {\mathbf{x}}
\def \xs {\x _\star}
\begin{align*}
Reg _T &\le \frac{T}{2\eta}\left( \Vert\x _1 - \xs\Vert^2 + \sum _{t=1}^T \Vert\x _t - \x _{t+1}\Vert^2 \right) = \frac{T}{2}\left( \frac{1}{\eta}\Vert\x _1 - \xs\Vert^2 + \eta\sum _{t=1}^T \Vert\nabla f _t(\x _t)\Vert^2 \right). \tag{5}
\end{align*}
$$

很显然，和梯度下降的式 [$(3)$](#regret-telescoping) 相比，式 [$(5)$](#gradient-ascent-regret) 中 $\eta$ 最多只能在括号里头做 trade-off，而括号外还有一个线性的 $T$，高下立判！

总而言之，在线梯度下降能够 work 就是因为它合理地利用了余弦定理形式的 telescoping 结构。你可能还会问：为什么得从上一步而不是上上步或者其他呢？从上面的推导可以看出，从上一步只是为了方便做 telescoping 的衔接。事实上，不一定得从上一步开始，只要你能构造出一条（或几条）的链条就行了，比如：
$$
\def \x {\mathbf{x}}
\x _1 \underset{-\nabla f _1(\x _1)}{\longrightarrow} \x _3 \underset{-\nabla f _3(\x _3)}{\longrightarrow} \cdots,\quad \x _2 \underset{-\nabla f _2(\x _2)}{\longrightarrow} \x _4 \underset{-\nabla f _4(\x _4)}{\longrightarrow} \cdots.
$$
更夸张地，你还可以：
$$
\def \x {\mathbf{x}}
\x _1 \underset{-\nabla f _1(\x _1)}{\longrightarrow} \x _T ,\quad \x _2 \underset{-\nabla f _2(\x _2)}{\longrightarrow} \x _3 \underset{-\nabla f _3(\x _3)}{\longrightarrow} \cdots \underset{-\nabla f _{T-2}(\x _{T-2})}{\longrightarrow} \x _{T-1}.
$$

不过其实就相当于把在线函数分为两组，跑两个 OGD 算法，本质上还是脱离不开“从上一步更新”的框架。能做成 telescoping 的关键还在于要从 $\mathbf{x} _t$ 用这点的梯度信息 $\nabla f _t(\mathbf{x} _t)$ 更新，剩下的就是一些变体了。