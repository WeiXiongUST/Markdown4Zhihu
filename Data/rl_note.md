# 1. 前言

我们考虑如下的一个 episodic MDP$(\mathcal{S}, \mathcal{A}, \mathbb{P}, r, H)$, 每一轮 Agent 从某个初始 状态 $x_1 \in \mathcal{S}$ 开始, 在每一轮:

- Agent 观察到当前的状态 $x_h \in \mathcal{S}$;
- Agent 采取动作 $a_h \in \mathcal{A}$;
- 环境产生一个奖励信号 $r_h(x_h,a_h) \in [0,1]$ 和下一个状态 $x_{h+1} \sim \mathbb{P}_h(\cdot|x_h,a_h)$ 并释放给用户;
- 整个游戏到达 $H+1$ 步的时候结束。

![image-20230609120838120](C:\Users\weixi\AppData\Roaming\Typora\typora-user-images\image-20230609120838120.png)

 我们考虑如下的Markov 策略 (policy), $\pi = \{\pi_h\}_{h=1}^H$, 其中
$$
\pi_h : \mathcal{S} \to \mathcal{A}.
$$
换句话说, 一个策略 就是一个从 状态 到动作空间的映射, 用来指导我们在不同 状态 下该采取什么样的动作。对于任意的一个策略, 我们定义如下的value function (价值函数), 来衡量一个策略 的好坏
$$
V_h^\pi(x) = \mathbb{E}_{\pi} [\sum_{h'=h}^H r_{h'}|x_h=x]\\       
        Q_h^\pi(x, a) = \mathbb{E}_{\pi} [\sum_{h'=h}^H r_{h'}|x_h=x, a_h=a].
$$
其中的V function 的期望应当被理解为, $a_h \sim \pi_h(\cdot|x_h=x), x_{h+1} \sim \mathbb{P}_h(\cdot|x_h,a_h), a_{h+1} \sim \pi_{h+1}(\cdot|x_{h+1})...$, 也就是我们不停采取 策略 $\pi$, 以及MDP 本身概率转移给出的随机性。我们的目的是学到最优策略 $\pi^*$, 其满足
$$
Q^{\pi^*}_h(x,a) = Q^*_h(x,a) := \sup_{\pi} Q^{\pi}_h(x,a)\\
V^{\pi^*}_h(x) = V^*_h(x) := \sup_{\pi} V^{\pi}_h(x)\\
\forall (h,x,a) \in [H] \times \mathcal{S} \times \mathcal{A}.
$$
注意这里的策略比较范围包含了随机性策略和一般的依赖于整个历史的策略, 但是我们知道对于MDP而言, 存在一个确定性且Markov 的策略满足以上条件。我们知道下列Bellman optimality equation 总是成立:
$$
Q^*_h(x,a) = (\mathcal{T}_h Q^*_{h+1})(x,a) := r_h(x,a) + \mathbb{E}_{x' \sim \mathbb{P}_h(\cdot|x,a)} V^*_{h+1}(x');\\
     V^*_{h+1}(x') = \max_{a' \in \mathcal{A}}Q_{h+1}^*(x',a').
$$


我们考虑如下的遗憾 (regret) 最小化问题, 假定我们总共有 $T$ 轮游戏, 且我们执行的策略序列为 $\pi_1, \cdots, \pi_T$, 则
$$
\mathrm{Reg}(T) := \sum_{t=1}^T V^*_1(x_1) - V^{\pi_t}_1(x_1).
$$
我们希望$\mathrm{Reg}(T)$ 是 $T$ 的低阶项, 此时我们有
$$
V^*_1(x_1) - \frac{\sum_{t=1}^T V^{\pi_t}_1(x_1)}{T} \to 0
$$
那么我们能够得到一个策略: 随机从 $\{\pi_1,\cdots,\pi_T\}$ 抽取一个并执行, 其接近于最优策略。



## 2 监督学习分析

我们首先分析一个监督学习的例子来介绍需要用到的数学工具。

**问题** 我们考虑一个假设空间 (Hypothesis space) $\mathcal{H}$ (例如线性函数空间, 神经网络参数空间), 为简单起见, 我们假设 $|\mathcal{H}| < \infty$。我们假定 未知有一个未知的分布 $P(X,Y)$, 对任意固定的假设 $f \in \mathcal{H}$, 它的泛化损失定义为
$$
L(f) = \mathbb{E}_{(X,Y) \sim P} \ell(f(X),Y),
$$
这里 $\ell(\cdot, \cdot)$ 是一个损失函数, 例如, $\ell(f(x),y) = \frac{1}{2}(f(x)-y)^2$。给定一个数据集 $\mathcal{D} = {(X_1,Y_1),\cdots,(X_n,Y_n)}$，我们还定义了*经验风险*为
$$
\hat{L}(f) = \frac{1}{n}\sum_{i=1}^n \ell(f(X_i),Y_i).
$$
显然经验损失是泛化损失的无偏估计量，在大数定律的某些温和条件下, 我们有
$$
\mathbb{E}_P \hat{L}(f) = L(f), \qquad \hat{L}(f) \to L(f), \qquad \text{as n tends to infinity.}
$$
假设我们通过最小化 $\hat{L}(f), f \in \mathcal{H}$（例如，我们运行SGD/Adam）找到一个 $\hat{f} \in \mathcal{H}$，并假设 $L(f)$ 的最小化者是 $f^* \in \mathcal{H}$。在这里，我们为了简单起见，假设 $0\leq \ell(\cdot, \cdot) \leq 1$。
$$
L(\hat{f}) - L(f^*) = \underbrace{\left(L(\hat{f}) - \hat{L}(\hat{f})\right)}_{A} + \underbrace{\left(\hat{L}(\hat{f}) - \hat{L}(f^*)\right)}_{B} + \underbrace{\left(\hat{L}(f^*) - L(f^*)\right)}_C
$$
首先，我们注意到 $B \leq 0$，因为 $\hat{f}$ 最小化了 $\hat{L}(\cdot)$, 所以$B$ 可以直接丢掉。对于$B$ 和 $C$, 我们注意到它们都是某个假设的泛化损失和经验损失之差, 我们希望用下面的引理来控制他们。

**Lemma 1** Hoeffding's inequality. 假设 $X_1, \cdots, X_n$ 独立且取值于 $[a,b]$, 则以概率至少$1-\delta$, 我们有
$$
|\bar{X}_n-\mathbb{E}X|\leq (b-a)\sqrt{\frac{\log(2/\delta)}{2n}} = O(\frac{b-a}{\sqrt{n}}).
$$
然而我们并不能直接利用上面的结论, 这是因为
$$
\hat{f} = \arg \min_{f \in \mathcal{H}} \hat{L}(f)
$$
是使用整个训练数据集产生的, 所以它依赖于整个训练数据, 导致 $\{\hat{f}(X_i)\}_{i=1}^n$ 不再是独立的随机变量。我们需要如下的 *Uniform convergence argument*:
$$
\begin{aligned}
L(\hat{f}) - L(f^*) &\leq \underbrace{\left(L(\hat{f}) - \hat{L}(\hat{f})\right)}_{A} + \underbrace{\left(\hat{L}(f^*) - L(f^*)\right)}_C\\
&\leq \sup_{f \in \mathcal{H}} |L(f) - \hat{L}(f)| + \left(\hat{L}(f^*) - L(f^*)\right)\\
&\leq 2\sup_{f \in \mathcal{H}} |L(f) - \hat{L}(f)|.
\end{aligned}
$$
在公式的最后, $f$ 是一个固定的假设, 因此它并不依赖于训练数据, 我们进一步有
$$
P(\sup_{f \in \mathcal{H}} |L(f) - \hat{L}(f)| > t) &= P(\exist f \in \mathcal{H} : |L(f) - \hat{L}(f)| > t) \\
&\leq \sum_{f \in \mathcal{H}} P(|L(f) - \hat{L}(f)| > t)
$$
其中最后一个不等式用到了union bound: 对任意事件$A, B$, 有
$$
P(A\cup B) \leq P(A) + P(B).
$$
我们设 $t = (b-a)\sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}$, 那么任意单个$f$的概率就为 $\delta/|\mathcal{H}|$, 从而
$$
P(\sup_{f \in \mathcal{H}} |L(f) - \hat{L}(f)| > (b-a)\sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}) \leq \delta.
$$
所以我们最终有, 以概率至少为 $1-\delta$, 
$$
L(\hat{f}) - L(f^*) \leq 2(b-a)\sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}
$$
这个分析方法叫*Uniform convergence*, 我们为了处理数据的依赖性, 付出了 $\log(|\mathcal{H}|)$ 的代价。如果我们进一步假定函数空间足够大, 使得 $f^* \in \mathcal{H}$ 满足$L(f^*) = 0$, 这个假定通常称为 *Realizability*。那么为了使得获取 $L(\hat{f}) \leq \epsilon$ , 我们实际上只需要 $O(\frac{(b-a)^2 \log (|\mathcal{H}|/\delta)}{\epsilon^2})$ 的样本。



换句话说, Realizability + bounded statistical complexity ($\log|\mathcal{H}|)$ 就能保证上面这个问题是可以解的。对于一般的无穷假设空间, 我们可以用标准的离散化技巧以及 covering number 来分析。



## 3 带函数近似的强化学习



### 3.1 Motivation

我们首先考虑 tabular MDP, 也就是 $|\mathcal{S}|, |\mathcal{A}| $ 都比较小的情况。这个时候我们可以用一张表 (table) 去存下整个Q-value, 所以也叫tabular case。这是最简单的一种MDP, 也可以说是最困难的一种MDP, 这是因为我们并不假设不同的状态-动作之间有任何潜在的结构使得它们之间能够泛化 (也就是知道 $s,a$ 的情况就能推断 $s',a'$)。



这个问题可以说已经被很好的解决了, 因为

- 我们已经有算法可以达到遗憾上界: $O(\sqrt{H^2 \textcolor{red}{S}AT})$ [AOM17, ZZJ20];
- 我们可以证明任何算法在最差情况下都有遗憾下界: $O(\sqrt{H^2 \textcolor{red}{S}AT})$。

我们可以看到tabular MDP 的下界依赖于状态空间大小, 但是现代的一些问题, 例如象棋有 $S=10^{40}$, 围棋有 $S^{170}$, 因此tabular的数学描述并不足以解释现实DRL的成功。我们的解决办法是考虑带有函数近似的RL问题, 也就是说, 我们用一个抽象的$|\mathcal{H}|$ 去近似:

1. 价值函数
2. MDP 模型
3. 策略

例如 DQN 会使用神经网络去模拟 $Q^*$, 此时, 神经网络的参数空间就构成了我们的假设空间 (当然, 我们的理论其实不咋能搞定神经网络, 所以就是一个例子 hh)

![image-20230609134522573](C:\Users\weixi\AppData\Roaming\Typora\typora-user-images\image-20230609134522573.png)

在每一轮 $t$, 我们会使用当前的某个假设 $f^t$, 并使用它去采样 ($\pi_{f^t}$)。



**线性函数近似** 一个最简单的情况, 我们假设我们可以抽象状态-动作对为一个$d$-维的向量, 并且 $Q^*$ 是线性于这样的一个特征向量的:$\exists$ $\theta_h \in R^d$ with $\norm{\theta^*_h} \leq B$ and a feature map $\phi:\mathcal{S} \times \mathcal{A} \to R^d$:
$$
Q^*_h(x,a) := \langle{\phi(x,a), \theta^*_h}\rangle, \qquad \forall h \in [H].
$$
我们可以用 $\mathcal{H}_h := \{\theta_h: \norm{\theta}_h \leq B\}$ 来近似这个问题, 这样 $\theta^* \in \mathcal{H}$, 同时线性空间的covering number 大约是$\tilde{O}(d)$, 一个很自然的问题是, 这时候RL的问题是不是可以解的 (因为监督学习这个情况下是可以解的)。下面这个结果告诉我们, 这些条件并不是充分的。



**Proposition 1** Linear-realizability is not sufficient [WWK21].  There exists an MDP with feature map $\phi$ that satisfies the linear approximation condition but any algorithms must have
$$
\mathbb{E} \mathrm{Reg}(T) \gtrsim \min\{2^{\Omega(d)}, 2^{\Omega(H)}\}.
$$
显然一个指数依赖于特征维度和游戏长度的结果不是我们想要的。



**挑战** 我们注意到, 监督学习和强化学习在以下方面是不一样的

- 监督学习: $\{x_i, y_i\}_{i=1}^n$ i.i.d. from a static distribution $\mathcal{D}_{\mathrm{data}}$;
- 强化学习: $\zeta_1 \sim \mathcal{D}_{\pi_{f^1}}, \zeta_2 \sim \mathcal{D}_{\pi_{f^2}}, \cdots, \zeta_T \sim \mathcal{D}_{\pi_{f^T}}$, distribution shifts all the time!

因此, 监督学习中能得到的泛化保证 (同分布假设!) 在强化学习中不再成立, 也就是说, 由于分布不同, 
$$
\mathcal{D}_1, \cdots, \mathcal{D_{t-1}} \overset{?}{\longrightarrow} \mathcal{D}_t
$$
我们需要一个不同的在线学习情况下的泛化保证: 这种泛化是incremental的, 并且能够处理distribution shift 的问题。



## 3.2 线性空间的泛化是有限的

一个粗糙的直觉是, 当我们考虑的问题的某些复杂度是有限的, 它不可能无限的去进行泛化 (也就是第$t$轮的分布和前面的历史分布差异极大), 我们希望找到这样一个描述这种泛化性能的度量。我们首先考虑$d$-维的线性空间。假定$\phi: \mathcal{Z} \to R^d$, 
$$
\mathcal{H} = \{f(\cdot)= \phi(\cdot)^\top \theta_f: \norm{\theta_f} \leq 1\}
$$
给定历史序列$\{f_1,g_1,z_1,\cdots, f_{t-1}, g_{t-1}, z_{t-1}\}$, 我们关心 $|f_t(z_t) - g_t(z_t)|$, 一般来说实际的分析里其中一个函数我们会取成最小化经验损失的假设, 实际上我们要求 $f-g$ 有线性结构就可以了。我们作如下分析 (类似的分析技术在contextual bandit 的文献里已经广泛存在了): 令 $\Sigma_t := \lambda I + \sum_{s=1}^{t-1} \phi(z_s)\phi(z_s)^\top$, 我们有
$$
|f(z_t) - g(z_t)|^2 &= {|\langle{\phi(z_t), \theta_f-\theta_g}\rangle|^2} \leq {\norm{\phi(z_t)}^2_{\Sigma_t^{-1}} \norm{\theta_f-\theta_g}^2_{\Sigma_t}}\\
    &\leq \norm{\phi(z_t)}^2_{\Sigma_t^{-1}}\Big(\lambda + \sum_{s=1}^{t-1} |f(z_s)-g(z_s)|^2\Big),
$$
 其中不等号用了Cauchy-Schwarz 不等式 $a^\top b = (\Sigma_{t}^{-1/2} a)^\top (\Sigma_t^{1/2}b) \leq \norm{\Sigma_{t}^{-1/2} a} \norm{\Sigma_t^{1/2}b}$ 且 $\norm{b}_{\Sigma_t} := \sqrt{b^\top \Sigma_t b}$. 换言之, 对任意的 $f,g$, 它在新的 $z_t$ 上的prediction error, 可以用带 $\lambda$ 正则的历史损失来控制, 但是会被 $\norm{\phi(z_t)}^2_{\Sigma_t^{-1}}$ 放大 (文献中叫 potential)。对于线性空间而言, 泛化是有限的, 具体的说, 我们有以下结论:

**Proposition 2 [Elliptical potential lemma]** 考虑 $\{\phi(z_t)\}_{t=1}^T$且 $\norm{\phi(z_t)} \leq L$对所有$t$成立, 则  $\norm{\phi(z_t)}_{\Sigma_t^{-1}}^2 > 1$ 发生最多 
$$
\frac{3d}{\log2}\log\big(1+\frac{L^2}{\lambda \log 2}\big) = \tilde{O}(d).
$$
此外, 我们有 
$$
\sum_{i = 1}^T \min \{1, \|x_i\|_{\Sigma_i^{-1}}^2 \} \le 2 \log \Big( \frac{ \det(\Sigma_{T+1})}{\det(\Sigma_1)} \Big) = \tilde{O}(d).
$$
直觉上说, 对任意的 $\{f_t,g_t,z_t\}_{t=1}^T$, 在新样本上prediction error 超出 historical error 太多的情况不会出现太多次, 这恰恰就是我们想要的在线学习情况下的泛化保证。具体地说, 我们有下面的结果:

**Lemma 1** Exploitation is save for linear model 
 We consider $\mathcal{H} = \{f(\cdot)= \phi(\cdot)^\top \theta_f: \norm{\theta_f} \leq 1\}$ where $\phi(z) \in R^d$ and $\norm{\phi(z)} \leq 1$ for all $z \in \mathcal{Z}$. For any sequence of $\{f_t,g_t,z_t\}_{t=1}^T$, we have
$$
\sum_{t=1}^T \underbrace{|f_t(z_t) - g_t(z_t)|}_{\text{Prediction error.}} \leq \tilde{O}\Big(\Big[d \cdot \sum_{t=1}^T \underbrace{\Big[\lambda + \sum_{s=1}^{t-1} (f_t(z_s) - g_t(z_s))^2\Big]}_{\text{Regularized historical error.}}\Big]^{1/2}\Big).
$$
**Proof**

令$\mathbf{1}(\cdot)$ 为事件的indicator 函数, 我们有
$$
\small
    \begin{aligned}
        \sum_{t=1}^T |f_t(z_t) - g_t(z_t)| &= \sum_{t=1}^T |f_t(z_t) - g_t(z_t)| \{\mathbf{1}(\norm{\phi(z_t)}_{\Sigma_t^{-1}}\leq 1)+ \mathbf{1}(\norm{\phi(z_t)}_{\Sigma_t^{-1}} > 1) \}\\
        &\leq \sum_{t=1}^T \min\{\norm{\phi(z_t)}_{\Sigma_t^{-1}}, 1\} \norm{\theta_{f^t}-\theta_{g^t}}_{\Sigma_t} + \sum_{t=1}^T\mathbf{1}(\norm{\phi(z_t)}_{\Sigma_t^{-1}} > 1) \\
        &\lesssim \sqrt{\sum_{t=1}^T \min\{\norm{\phi(z_t)}^2_{\Sigma_t^{-1}}, 1\}} \sqrt{\sum_{t=1}^T \norm{\theta_{f^t} - \theta_{g^t}}^2_{\Sigma_t}} + d \log\big(1+\frac{1}{\lambda}\big)\\
        &\leq \tilde{O}\Big(\Big[d \cdot \sum_{t=1}^T \Big[\lambda + \sum_{s=1}^{t-1} (f_t(z_s) - g_t(z_s))^2\Big]\Big]^{1/2}\Big),
    \end{aligned}
$$
其中第二个不等式我们略去了一些常数并用了 Proposition 2 的第一个结论, 第三个不等式用了 Proposition 2 的第二个结论。

**End of Proof**



我们希望把 Lemma 1 中的想法抽象出来, 但是在此之前, 我们需要明确在RL 问题中的 prediction errror。



## 3.3 遗憾分解与Bellman Error

假定我们有一个假设序列 $f$, 且它能导出一个价值函数对 $(Q_{h,f}, V_{h,f})_{h=1}^H$, 例如, 如果$f_h: \mathcal{S} \times \mathcal{A} \to [0,H]$, $f_h$ 本身就是Q-value 而对动作空间取 max 之后就是V-value; 而如果 $f = (\mathbb{P}_{h,f}, r_{h,f})_{h=1}^H$, 我们可以用$f$ 这个MDP model 下的最优Q/V 价值函数作为它导出的价值函数对。我们总是假定 $V_{h,f}(x) = \max_{a' \in \mathcal{A}} Q_{h,f}(x, a')$。 我们定义其Bellman error 为
$$
\mathcal{E}_h(f, x,a) := Q_{h,f}(x,a) - (\mathcal{T}_h V_{h+1,f})(x,a).
$$
那么我们有: 
$$
\begin{aligned}
    \mathrm{Reg}(T) = \sum_{t=1}^T V_{1}^{*}\left(x_{1}\right)-V_{1}^{\pi_{f^t}}\left(x_1\right)&=\sum_{t=1}^T [V_{1,f^t} - V_1^{\pi_{f^t}}] + \sum_{t=1}^T [V_1^* - V_{1,f^t}]\\
    &\leq \underbrace{\sum_{t=1}^T\sum_{h=1}^{H} \mathbb{E}_{\pi_{f^t}}\left[\mathcal{E}_{h}\left(f^t, x_{h}^t, a_{h}^t\right)\right]}_{\mathrm{(I)}} - \underbrace{\sum_{t=1}^T\sum_{h=1}^{H} \mathbb{E}_{\pi^*} \left[\mathcal{E}_{h}\left(f^t, x_{h}^t, a_{h}^t\right)\right]}_{\mathrm{(II)}}.
    \end{aligned}
$$
可以看到, 遗憾可以被分解为两个和 Bellman error 相关的项, 主要差别在于我们取期望的假设。我们在这里介绍一般考虑的两个setting以及对应的假设。

**Online learning** 我们从头开始, 每一轮选定$f^t \in \mathcal{H}$, 并使用$\pi_{f^t}$ 收集数据, 总共进行T轮。

此时第一项是可能被控制的, 但是需要我们对$\mathcal{H}$ 和MDP 之间做一些假设, 使得能够从收集到的数据 ($f^1, \cdots,f^{t-1}$) 来推断 $f^t$ 导出的分布 (可能需要一些针对性的算法设计); 第二项是难以控制的, 因为我们完全不知道 $\pi^*$ 是什么, 无法针对性的作算法设计。

因此, 对于online learning, 我们通常会采取 optimistic 的算法设计, 也就是始终让 $V_{1,f^t} > V^*_1$, 从而直接控制掉第二项, 此时我们的研究对象就从 $\mathrm{Reg}(T)$ 转向了
$$
\sum_{t=1}^T [V_{1,f^t} - V_1^{\pi_{f^t}}] = \sum_{t=1}^T\sum_{h=1}^{H} \mathbb{E}_{\pi_{f^t}}\left[\mathcal{E}_{h}\left(f^t, x_{h}^t, a_{h}^t\right)\right].
$$
也就是 *out-of-sample* 的Bellman error。

**Offline learning** 我们被给定一个数据集 $\{\zeta^t = (x_1,a_1,r_1,\cdots,x_H,a_H,r_H)\}_{t=1}^T$, 并需要利用这个数据集学习一个策略。

我们事实上只$T=1$, 对于任意单个的 $f$, 我们在不做一些比较强的假设的情况下, 第一项是比较难控制的, 因为我们并不知道MDP 的dynamic, 从而不知道期望如何计算, 相反, 我们可以固定单个 $f^*$ 作一些数据集上的假设 (文献里称为coverage, 也就是说数据集分布能够 ``cover'' 住 $\pi^*$ 导出的分布), 此时第二项可以控制。

因此, 对于offline learning, 我们通常会采取 pessimistic 的算法设计, 也就是始终让 $V_{1,f^t} \leq V^{\pi_{f^t}}_1$, 从而直接控制掉第一项, 关于 offline 情况pessimism的设计, 可以参考[JYW21]的仔细讨论, 这里就不详细写了。

## 3.4 Eluder Coefficient

针对上一轮确定的prediction error, 我们可以扩展线性模型中得到的直觉为下面的复杂度。

**Definition 1** Generalized Eluder coefficient [DMZZ21, ZXZ+22]

给定一个MDP和一个假设空间 $\mathcal{H}$, eluder coefficient $d(\epsilon)$ 定义为最小的  $d$ ($d\geq 0$) 使得对任意的假设序列 $\{f^t \in \mathcal{H}\}_{t=1}^T$, 我们有: 
$$
\begin{aligned}
    &\sum_{t = 1}^T \underbrace{V_{1,f^t} - V_1^{\pi_{f^t}}}_{\displaystyle \text{\textcolor{red}{prediction error}}}= \sum_{t=1}^T \sum_{h=1}^H \mathbb{E}_{\pi_{f^t}} \mathcal{E}_h(f^t,x_h,a_h)\\
    &\qquad \le \bigg [ \underbrace{d(\epsilon)}_{\displaystyle \text{\textcolor{violet}{cost of generalization}}} \sum_{h=1}^H \sum_{t=1}^T \underbrace{\sum_{s=1}^{t-1}\Big( \mathbb{E}_{{\pi}_{f^s}} \mathcal{E}_h(f^t, x_h, a_h) \Big)^2}_{\displaystyle \text{\textcolor{cyan}{training error}}} \bigg]^{1/2} + \underbrace{ 2\min\{Hd, H^2T\} + \epsilon B^\dagger T}_{ \displaystyle \text{burn-in cost}},
\end{aligned}
$$
这里 $B^\dagger>0$ 是一个和问题有关的常数。更一般的, 上面的讨论确定了我们的prediction error, 但是我们可以有更自由的 training error 的选择, 这让我们可以处理更多的问题, 最终有:
$$
\begin{aligned}
    \sum_{t = 1}^T {V_{1,f^t} - V_1^{\pi_{f^t}}}\lesssim \bigg [ \underbrace{d(\epsilon)}_{\displaystyle \text{\textcolor{violet}{cost of generalization}}} \sum_{h=1}^H \sum_{t=1}^T \underbrace{\sum_{s=1}^{t-1} \ell_h^s(f^t)}_{\displaystyle \text{\textcolor{cyan}{training error}}} \bigg]^{1/2},
\end{aligned}
$$
其中$\ell_h^s(\cdot): \mathcal{H} \to R^+$ and $\ell_h^s(f^*) = 0$。为简单, 我们省却了 burn-in cost, 事实上, 它们通常都是 non-dominating的。相比起之前线性空间的例子, $\ell_h^s(\cdot)$ 的定义中通常还包含了一个期望, 这使得我们的结构性假设隐式的也做在了MDP的dynamic上，从而使得我们可以包含更多的例子。



利用与线性空间中类似的分析技术, 我们可以直接验证以下两种被广泛考虑的线性近似的 MDP, 其证明的过程可以总结为, 证明 the class of averaged Bellman error is linear。



**Linear MDP**: TL;DR: Both the reward and transition kernel are linear in the feature map.

MDP$(\mathcal{S}, \mathcal{A}, H, \mathbb{P}, r)$ is a linear MDP with a (known) feature map $\phi:\mathcal{S}\times \mathcal{A} \to R^d$, if for any $h \in [H]$, there exist $d$ unknown signed measures ${\mu}_{h}=(\mu_{h}^{(1)}, \cdots, \mu_{h}^{(d)})$ over $\mathcal{S}$ and an unknown vector $\theta_h \in R^d$, such that for any $(x,a) \in \mathcal{S} \times \mathcal{A}$, we have 
$$
\mathbb{P}_{h}(\cdot \mid x, a)=\left\langle{\phi}(x, a), {\mu}_{h}(\cdot)\right\rangle, \qquad r_{h}(x, a)=\left\langle{\phi}(x, a), {\theta}_{h}\right\rangle.
$$
Without loss of generality, we assume that $\norm{\phi(x,a)}\leq 1$ for all $(x,a) \in \mathcal{S}\times \mathcal{A}$, and $\max\{\norm{\mu_h(\mathcal{S})}, \norm{\theta_h}\} \leq \sqrt{d}$ for all $h \in [H]$.

对于 linear MDP, 如果我们取 $\mathcal{H}_h = \{Q_{h,f}(\cdot,\cdot) = \phi(\cdot,\cdot)^\top \theta_{h,f}: \|\theta_{h,f}\| \leq \sqrt{d} H\}$, 对于 $\ell_h^s(f) = \big(\mathbb{E}_{\pi_{f^s}} \mathcal{E}_h(f^t, x_h, a_h) \Big)^2$, 我们有
$$
d(\epsilon) = O\big(Hd\log(1+\frac{T}{\epsilon})\big).
$$
**Proof**

首先我们注意到, 由于奖励函数和转移矩阵都线性于特征向量, 所以任何函数经过一次Bellman update, 都是线性的:
$$
\mathcal{T}_hV(x,a) &= r_h(x,a) + (\mathbb{P}_hV)(x,a) = \phi(x,a)^\top \theta_h + \int_{\mathcal{S}} V(x_{h+1}) \langle{\phi(x,a), d\mu_h(x_{h+1})}\rangle\\
 &= \Big\langle\phi(x,a), \theta_h + \int_{\mathcal{S}}V(x_{h+1}) d\mu_h(x_{h+1})\Big\rangle := \langle{\phi(x,a), w_h}\rangle.
$$
这说明了$Q^*_h$是线性的, 因为 $Q^*_h = \mathcal{T}_hV^*_{h+1}$, 利用$V^*_{h+1}\in [0, H-1]$, 以及定义中的正则范围, 我们有 $Q^* \in \mathcal{H}$, 以及
$$
\mathbb{E}_{\pi_f} \mathcal{E}_h(f,x_h,a_h) = \mathbb{E}_{\pi_f}[Q_{h,f}(x_h,a_h) - \mathcal{T}_h V_{h+1,f}(x_h,a_h)]= \langle\mathbb{E}_{\pi_f} \phi(x_h,a_h), \theta_{h,f} - w_{h,f}\rangle := \langle X_h(f), \theta_{h,f} - w_{h,f}\rangle.
$$
也就是说, 任意平均Bellman error 也是线性的。之后只要考虑 $\Sigma_{t;h} = \lambda I + \sum_{s=1}^{t-1} X_h(f^s) X_h(f^s)^\top$, 并重复线性空间里的分析就可以得到结果。

**End of Proof**

可以看到, 相比于 $Q^*$ 是线性的 (我们强调它是不充分的条件)，我们这里额外得到了 任意$V: \mathcal{S} \to R$ 的一次Bellman update 也落回到线性空间里, 特别的, 任意$V_{h+1, f}(x) = \max_a Q_{h+1,f}(x,a)$ 的Bellman update 仍然是线性的, 换句话说, 这个假设空间是封闭于 Bellman update 的。事实上, 这已经足够推出它的泛化是有限的 (eluder coefficient 为 $\tilde{O}(d)$), 而 linear MDP 是更强的一个条件。



**Linear Mixture MDP**: TL;DR: The transition kernel is a linear mixture of a number of basis kernels. 

对于Liner mixture MDP, 存在已知的两个特征 $\phi: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to R^d$ and $\psi: \mathcal{S} \times \mathcal{A} \to R^d$;使得对任意的$h \in [H]$ 和 $(x,a,x') \in \mathcal{S} \times \mathcal{A} \times \mathcal{S}$, 我们有
$$
\mathbb{P}_{h}(x^{\prime} \mid x, a)=\langle\theta_{h}^{*}, \phi(x, a, x^{\prime})\rangle, \qquad  r_h(x, a)=\langle\theta_{h}^{*}, \psi(x, a)\rangle.
$$
我们假定 $\norm{\theta_h^*} \leq B$ 且 $B > 0$.

对于 linear mixture MDP, 如果我们取$\mathcal{H} = \{f = (\theta_{1,f}, \cdots, \theta_{H,f}): \forall h \in [H], \norm{\theta_{h,f}} \leq B\}$, then it has an eluder coefficient of $d(\epsilon) = \tilde{O}\big(Hd\big)$ with
$$
\begin{aligned}
\ell_h^s(f) &= \mathbb{E}_{\pi_{f^s}}\Big[r_{h,f}(x_h,a_h) - r_{h,f^*}(x_h,a_h) \\
&\qquad + \mathbb{E}_{x_{h+1} \sim \mathbb{P}_{h,f}(\cdot|x_h,a_h)} V_{h+1,f^s}(x_{h+1}) - \mathbb{E}_{x_{h+1} \sim \mathbb{P}_{h,f^*}(\cdot|x_h,a_h)} V_{h+1,f^s}(x_{h+1}) \Big].
    \end{aligned}
$$
我们不作具体的证明, 因为证明技术是类似的, 只不过我们去验证线性的方式不太一样。



更一般的, 我们容易验证 Bellman rank [JKA+17], witness rank [SJK+19], 以及bilinear class [DKL+21]满足类似的条件, 因为它们假设了 average Bellman class 有一个bilinear 的结构, 从而可以用上述的分析技术去验证。



## 4 算法设计: Optimization with Feel-good Modification

当一个问题有较低的 eluder coefficient, 在optimism达成的情况下, 我们只需要关注 in-sample 的历史损失即可。但是我们注意到, $\ell_h^s(f)$ 通常包含有期望, 因此并不能直接从数据集中得到。我们需要构造恰当的估计量来对它们进行估计。 

 ### 4.1 历史损失估计

我们定义如下的估计量。

**Definition 2** Loss estimator

我们考虑一个更一般性的采样策略: 我们将$T$ 轮游戏分成 $K$个阶段, 每个阶段$k$ 我们独立地采样 $m$ 条路径 $\{\zeta_{i,h}^k\}_{i=1}^m$, 那么$T = mK$。我们假定我们有如下的一个估计量: $L_h^{1:k-1}(\cdot):\mathcal{H} \to R$, 它只依赖于历史$\{f^1,(\zeta^1_{i,h})_{i=1}^m,f^2,(\zeta^2_{i,h})_{i=1}^m,\cdots,f^{k-1},(\zeta^{k-1}_{i,h})_{i=1}^m\}$, 同时, 以概率至少$1-\delta$, 对任意的$(k,h,f) \in [K] \times [H] \times \mathcal{H}$, 它满足
$$
    \begin{aligned}
    \sum_{s=1}^{k-1} \ell_h^s(f) &\leq L_h^{1:k-1}(f) + \Delta_h^k,\\
        L_h^{1:k-1}(f^*) &\leq \Delta_h^k.
    \end{aligned}
$$
当$|\ell_h^s| \leq b$ 有界的时候, 我们可以令这里 $\Delta_h^k = (k-1)b$ 且 $L_h^{1:k-1} = 0$, 但是我们通常可以有更好的估计量, 这些估计量的构造基本上是一个关于concentration inequality 的练习。



我们首先考虑value-based 的情况, 也就是 $\ell_h^s(f) =\big(\mathbb{E}_{\pi_{f^s}} \mathcal{E}_h(f, x_h, a_h)\big)^2$, 为了估计它, 我们需要解决所谓的double-sampling issue, 简单来说, 就是
$$
\mathbb{E} \sum_{s=1}^{k-1} X_s^2 = \underbrace{\sum_{s=1}^{k-1} (\mathbb{E} X_s)^2}_{\text{Goal}} + \underbrace{\sum_{s=1}^{k-1} \sigma_s^2}_{\text{Sampling variance}}.
$$
也就是将期望从平方外拿到平方里, 会带来一个线性累计的方差项, 这是我们所不能接受的。为了解决这个问题, 一个很自然的想法就是使用 sample-mean 来降低方差。

**Trajectory Average with realizability.**
We can independently sample $m$ trajectories $\{\zeta^k_{i,h}\}_{i=1}^m$ by following $\pi_{f^k}$ and take
$$
L_h^{1:k-1}(f) = \sum_{s=1}^{k-1} L_h^s(f) :=  2 \sum_{s=1}^{k-1} \Big[\underbrace{\frac{1}{m} \sum_{i=1}^m \Big(Q_{h,f}(x^s_{i,h},a^s_{i,h}) - r^s_{i,h}  - V_{h+1,f}(x^s_{i,h+1})\Big)}_{\displaystyle \text{Sample mean.}}\Big]^2,
$$
则我们有 $\Delta_h^{k} = \frac{4(k-1) H^2 \iota}{m}$, and $\iota = O(\log(KH |\mathcal{H}|/ \delta))$. 

**Proof**

我们记 $\epsilon_{h}^s(f) = \frac{1}{m} \sum_{i=1}^m (Q_{h,f}(x^s_{i,h},a^s_{i,h}) - r^s_{i,h}  - V_{h+1,f}(x^s_{i,h+1}))$。对任意固定的$(s,h,f) \in [K] \times [H] \times \mathcal{H}$, 我们使用Azuma-Hoeffding 不等式, 则至少以概率 $1-\delta/(KH|\mathcal{H}|)$, 我们有
$$
\Big|\epsilon_{h}^s(f) - \mathbb{E}_{\pi_{f^s}}\mathcal{E}_h(f, x_h, a_h)\Big| \leq H\sqrt{\frac{2\log(KH|\mathcal{H}|/\delta)}{m}}.
$$

​    根据 $(a+b)^2 \leq 2a^2 + 2b^2$, 我们有
$$
\big(\mathbb{E}_{\pi_{f^s}} \mathcal{E}_h(f, x_h, a_h)\big)^2 \leq 2\big({\epsilon}_h^s(f)\big)^2 + \frac{4H^2\log(KH|\mathcal{H}|/\delta)}{m}.
$$
​    对 $(s,h,f) \in [K] \times [H] \times \mathcal{H}$ 取一个 union bound即有, 以概率至少 $1-\delta$, 
$$
\sum_{s=1}^{k-1} \ell_h^s(f) \leq L_h^{1:k-1}(f)  + \frac{4(k-1)H^2 \log(KH|\mathcal{H}|/\delta)}{m}.
$$
**End of Proof**

注意, 上述的估计量应当被理解为:
$$
m \cdot \sum_{s=1}^{k-1} \ell_h^s(f) \leq m \cdot L_h^{1:k-1}(f) + 4(k-1)H^2 \cdot \log(KH|\mathcal{H}|/\delta).
$$
因此它事实上是sub-optimal的。如果有额外的假设条件, 我们可以得到更sharp 的估计量。



**Minimax Formulation with Bellman completeness**

我们额外假设, $\mathcal{T}_h V_{h+1,f} \in \mathcal{Q}_h$, 也就是说假设空间关于Bellman operator 是封闭的。此时, 我们可以设$m=1$ 且 采用下面的估计量
$$
L_h^s(f) = \big(Q_{h,f}(x_h^s,a_h^s) - r_h^s - V_{h+1,f}(x_{h+1}^s)\big)^2 - \underbrace{\inf_{f'_h \in \mathcal{H}_h} \big(
Q_{h,f'}(x_h^s,a_h^s) - r_h^s - V_{h+1,f}(x_{h+1}^s)\big)^2}_{\displaystyle \text{Approximate the sampling variance.}},
$$
with $\Delta_h^t = O(H^2 \iota)$ and $\iota=O(\log(H|\mathcal{H}|T/\delta))$. 这个方法称为 minimax formulation 是因为我们可以重写为:
$$
\min_{f \in \mathcal{H}} \max_{f' \in \mathcal{H}} \sum_{h=1}^H\Big[\big(Q_{h,f}(x_h^s,a_h^s) - r_h^s - V_{h+1,f}(x_{h+1}^s)\big)^2 - \big(
Q_{h,f'}(x_h^s,a_h^s) - r_h^s - V_{h+1,f}(x_{h+1}^s)\big)^2\Big].
$$
引入的第二项能够用来cancel 采样的方差, 但是另一方面, 在分析中, 我们需要使用 Bellman completeness 条件来控制这一项。这里的 $\Delta_h^t$ 并不会随着 $t$ 线性增长, 因此比起使用样本均值要更加高效。然而, 我们注意到从$V_{H+1}=0$出发, 并利用Bellman completeness, 我们可以推导出 $Q^* \in \mathcal{H}$, 换言之 completeness 比起realizability 是更强的假设。并且这个假设本身是 non-monotone的, 也就是说, 即使我们有一个满足这个条件的假设空间 $\mathcal{G}$, 我们增加一个新的假设后, $\mathcal{G} \cup \{g\}$ 很可能不再满足这个假设, 这个证明复杂一些, 我们就不详细写了。



我们也可以考虑model-based 的情况, 此时, 我们有: $\ell_h^s(f) = \mathbb{E}_{\pi_{f^s}} D_{\mathrm{H}}^2 \big(\mathbb{P}_{h,f}(\cdot \mid x_h,a_h), \mathbb{P}_{h,f^*}(\cdot \mid x_h,a_h) \big)$。这里的$D_{\mathrm{H}}^2(\cdot, \cdot)$ 是Hellinger distance, 定义为
$$
D_{\mathrm{H}}^2(P, Q) = \frac{1}{2} \int_{\mathcal{X}} \big( \sqrt{P(x)} - \sqrt{Q(x)}\big)^2 dx = 1 - \int_\mathcal{X} \sqrt{P(x) Q(x)} dx, 
$$
这里 $P$ 和 $Q$ 是 pmf 或者 pdf。



**Model-based realizability**

此时我们可以取 $L_h^{1:t}(f) := \sum_{s = 1}^t L_h^s(f) :=\frac{1}{2}\sum_{s = 1}^t -\log \mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)$。 但是这样一个估计量并不能给我们所需要的条件, 特别$L_h^{1:t}(f^*)$ 并不会比较小。我们的解决办法是给所有的假设都减掉相同的一个 baseline 
$$
\tilde{L}_h^s(f) =  -\frac{1}{2}\log \mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s) - \underbrace{[-\frac{1}{2}\log \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)]}_{\displaystyle \text{Baseline.}}
$$
注意第二项事实上我们的算法是无法获得的, 但是由于我们对所有假设减掉相同的一个baseline, 使用$L_h^s(\cdot)$ 在理论上等价于使用 $\tilde{L}_h^s(\cdot)$, 所以我们可以拿它来做理论分析。此时我们有$\Delta_h^t = O(H^2 \iota)$ and $\iota=O(\log(H|\mathcal{H}|T/\delta))$. 

**Proof**

我们首先控制 moment-generating function。我们记 $\mathbb{E}_t = \mathbb{E}[\cdot|(f^1,\zeta^1, \cdots, \zeta^{t-1},f^t)]$, 也就是做第$t$轮路径的期望, 则有
$$
\begin{aligned}
    &\mathbb{E}\Big[\exp\Big(\frac{1}{2}\sum_{s=1}^t \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}\Big)\Big]\\
    &= \mathbb{E}\Big[\exp\Big(\frac{1}{2}\sum_{s=1}^{t-1} \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}\Big)\Big] \mathbb{E}_t \sqrt{\frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^t, a_{h}^t)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^t, a_{h}^t)}}\\
    &= \mathbb{E}\Big[\exp\Big(\frac{1}{2}\sum_{s=1}^{t-1} \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}\Big)\Big] \mathbb{E}_{\pi_{f^t} }\int_{x \in \mathcal{S}} \sqrt{\mathbb{P}_{h, f} (x \mid x_{h}, a_{h}) \cdot \mathbb{P}_{h, f^*} (x \mid x_{h}, a_{h})}\\
    &= \mathbb{E}\Big[\exp\Big(\frac{1}{2}\sum_{s=1}^{t-1} \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}\Big)\Big]\Big(1-\mathbb{E}_{\pi_{f^t} } D_{\mathrm{H}}^2 \big(\mathbb{P}_{h,f}(\cdot \mid x_h,a_h), \mathbb{P}_{h,f^*}(\cdot \mid x_h,a_h) \big)\Big)\\
    &= \cdots\\
    &= \prod_{s=1}^t \Big(1-\mathbb{E}_{\pi_{f^s} } D_{\mathrm{H}}^2 \big(\mathbb{P}_{h,f}(\cdot \mid x_h,a_h), \mathbb{P}_{h,f^*}(\cdot \mid x_h,a_h) \big)\Big).
     \end{aligned}
$$
我们需要如下的引理: 

**Lemma 2** Martingale Exponential Inequalities 

Consider a sequence of random functions $\xi_1(\mathcal{Z}_1), \cdots, \xi_t(\mathcal{Z}_t), \cdots$ with respect to filtration $\{\mathcal{F}_t\}$. We have for any $\delta \in (0,1)$ and $\lambda > 0$:
$$
\mathbb{P}\Big[\exists n > 0: - \sum_{i=1}^n \xi_i \geq \frac{\log(1/\delta)}{\lambda} + \frac{1}{\lambda} \sum_{i=1}^n \log \mathbb{E}_{Z_i^{(y)}} \exp(-\lambda \xi_i)\Big] \leq \delta,
$$
where $Z_t = (Z_t^{(x)}, Z_t^{(y)})$ and $\mathcal{Z}_t = (Z_1,\cdots,Z_t)$. See e.g., Theorem 13.2 of \cite{zhang2022mathematical} for proof.



利用以上的集中不等式, 对任意固定的 $h, f$, 我们有
$$
1-\frac{\delta}{H |\mathcal{H}|}&\leq \mathbb{P}\Big[\forall t > 0: \frac{1}{2}\sum_{s=1}^t \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)} \leq \log(H |\mathcal{H}|/\delta) \\
&\qquad + \sum_{s=1}^t \log \Big(1-\mathbb{E}_{\pi_{f^s} } D_{\mathrm{H}}^2 \big(\mathbb{P}_{h,f}(\cdot \mid x_h,a_h), \mathbb{P}_{h,f^*}(\cdot \mid x_h,a_h) \big)\Big]\\
&\leq \mathbb{P}\Big[\forall t > 0: \frac{1}{2}\sum_{s=1}^t \log \frac{\mathbb{P}_{h, f} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)}{ \mathbb{P}_{h, f^*} (x_{h+1}^s \mid x_{h}^s, a_{h}^s)} \leq \log(H |\mathcal{H}|/\delta) \\
&\qquad - \sum_{s=1}^t \mathbb{E}_{\pi_{f^s} } D_{\mathrm{H}}^2 \big(\mathbb{P}_{h,f}(\cdot \mid x_h,a_h), \mathbb{P}_{h,f^*}(\cdot \mid x_h,a_h)\big)\Big].
$$
最后加上一个union bound就完成了证明。

**End of Proof**

## 4.2 Optimistic Modification

在 3.3 节中, 我们提到我们需要解决 $V_1^*$ 与 $V_{1,f^t}$ 的不同, 事实上, 如果我们直接去最小化 4.1 节中构造的损失估计量:
$$
f^k = \arg \min_{f \in \mathcal{H}} L_h^{1:k-1}(f),
$$
通过GEC, 我们事实上最小化的是 $V_{1,f^k} - V^{\pi^{f^k}}_1$ 而不是我们想要最小化的 $\mathrm{Reg}(T)$。为了解决这个问题, 我们修改上述的目标函数如下:
$$
f^k = \arg \min_{f \in \mathcal{H}} [-V_{1,f}(x_1) + \eta \cdot L_h^{1:k-1}(f)],
$$
出于习惯, 我们等价的写为
$$
f^k = \arg \max_{f \in \mathcal{H}} [V_{1,f}(x_1) - \eta \cdot L_h^{1:k-1}(f)],
$$
其中$\eta > 0$ 是一个用来控制加入的偏置项重要性的参数。完整的算法如下: 

![image-20230610155752679](C:\Users\weixi\AppData\Roaming\Typora\typora-user-images\image-20230610155752679.png)

与传统 OFU-based 算法相比, 这里不需要显式的维护一个 version space (包含有近似取得最小损失估计量的那些假设), 然后在 version space 上做 constraint optimization 来获取最为 optimistic 的假设。因此, 在实践中, 这样的一个偏置项修改会更加容易实现, 感兴趣的可以参考 [LLX+23]。

理论上, 我们有如下的结论。

**Theorem 1**

假定 realizability 成立, 如果问题有 GEC $d(\epsilon)$ 且对应的损失估计量对应估计区间$\Delta_h^k$, 那么, 以概率至少$1-\delta$, 我们有
$$
\mathrm{Reg}(T) \leq 2\sqrt{2} m \sqrt{\underbrace{d(\epsilon)}_{\text{Cost of generalization.}} \underbrace{\sum_{h=1}^H \sum_{k=1}^K \Delta_h^{k}}_{\text{In-sample supervised guarantee.}}} + 2m \cdot \min\{Hd, H^2K\} + \epsilon B^\dagger T,
$$
其中 $T=mK$.

我们可以看到, 遗憾的上界此时由两部分决定, 一部分是来自于泛化的代价, 一部分是我们在收集到的数据上做supervised learning 所能获得的理论保证。

**Proof**

我们有:
$$
    \begin{aligned}
    &\sum_{k=1}^K V^*_1 - V_1^{\pi^k} := \sum_{k=1}^K [V_{1,f^k} -  V_1^{\pi^k} + V^*_1 - V_{1,f^k}]\\
    &\leq \sum_{k = 1}^K[V^*_1 - V_{1,f^k}] + \eta \sum_{h=1}^H \sum_{k=1}^K \Big( \sum_{s=1}^{k-1} \ell_h^s(f^k) \Big) + \underbrace{\frac{1}{\eta} \cdot d + 2\min\{Hd, H^2K\} + \epsilon B^\dagger K}_{{\Xi}} ~~~\text{//\textcolor{red}{Eluder}}\\
    &\leq \sum_{k = 1}^K V^*_1 + \textcolor{blue}{\sum_{k=1}^K \big[ - V_{1,f^k} + \eta \sum_{h=1}^H L_h^{1:k-1}(f^k)\big]} + \eta\sum_{h=1}^H\sum_{k=1}^K \Delta_h^{k} + \Xi~~~~~~~~~~\text{//\textcolor{red}{Loss estimator.}}\\
    &\leq \sum_{k=1}^K V_1^* + \textcolor{blue}{\sum_{k=1}^K\big[-V_1^* + \eta \sum_{h=1}^H L_h^{1:k-1}(f^*)\big]} + \eta\sum_{h=1}^H\sum_{k=1}^K \Delta_h^{k} + \Xi ~~~~~~~~~~~~~\text{//\textcolor{red}{Update rule.}}\\
    &\leq 2\eta \sum_{h=1}^H\sum_{k=1}^K \Delta_h^{k} + \frac{1}{\eta} \cdot d + 2\min\{Hd, H^2K\} + \epsilon B^\dagger K.
    \end{aligned} 
$$
对于最后一项, 我们针对$\eta > 0$ 最小化, 并在两边乘上 $m$ 即证。(我们一轮采了$m$条路径)

**End of Proof**



## 5 Eluder dimension

Eluder coefficient 和 Eluder dimension [RVR13] 密切相关, 并且有类似的动机, 我们首先介绍 Eluder dimension 的概念。

**$\epsilon$-dependence**
A point $z$ is $\epsilon$-dependent on a set $\{x_1,...,x_n\}$ with respect to $\mathcal{F}$ if any $f,g\in\mathcal{F}$ such that $\sqrt{\sum_{i=1}^n (f(x_i)-g(_i))^2}\le\epsilon$ satisfies $|f(z)-g(z)|\le\epsilon$.

简单来说, 对任意的两个函数$f,g$, 如果它们在已经观察到的$n$个点上都基本上一样, 那么它们在未知的$z$ 上的点的函数值也差不多, 那么这时候它们就是相关的。相反, 独立就是说, 存在两个函数 $f,g$, 即使它们在观察到的点上基本一样, 但是它们在新的点上仍然差的比较大, 例如下面的图中(来自 [JLM21] 的talk), 四个函数虽然在观察到的四个点$x_1,x_2,x_3,x_4$ 都差不多, 但是它们在$z$ 却有较大的差距。

![image-20230610182240047](C:\Users\weixi\AppData\Roaming\Typora\typora-user-images\image-20230610182240047.png)

**Eluder dimension**

For $\epsilon>0$ and a function class $\mathcal{F}$ defined on $\mathcal{X}$, the $\epsilon$-eluder dimension $\dim_E(\mathcal{F},\epsilon)$ is the length of the longest sequence of elements in $\mathcal{X}$ such that for some $\epsilon'\ge\epsilon$, each element is $\epsilon'$-independent of its predecessors.

也就是我们能找到独立的最长序列长度, 用我们之前泛化的例子来说, 就是我们一直选择在历史数据集上表现的比较好的 $f$, 但是我们在新的一轮游戏里遭受比较大损失的最多次数。



[RVR13] 证明了, 如果我们能保证 $\sum_{s=1}^{k-1}  f^k(z_s)^2 \leq \beta_{k}$ 对一个 **non-decreasing** 的$\beta_k$ 序列成立, 那么我们有
$$
\sum_{k=1}^K |f^k(z_k)| \leq \sqrt{d(\epsilon) \cdot \sum_{k=1}^K \beta_k} + \min \{K, d(\epsilon)\} + K\epsilon
$$
对于显式维护 version space 的算法, 以上的结论足够用来分析算法。然而, 对于我们上面的算法, 或者 Thompson sampling 算法 [DMZZ21], 由于我们对于历史损失没有一个 non-decreasing 的上界保证, 这个结果不足以来分析算法。



[DMZZ21] 发展了新的分析技术, 证明了$d_{GEC}(\epsilon) = \tilde{O}(d_{Eluder}(\epsilon))$, 从而使得任意在 Eluder dimension 框架下的问题也能在 Eluder coefficient 的框架下处理。此外, 我们可以找到例子 [XFB+22], 它满足 $d_{Eluder} = \Omega(T^{1/3})$ 且$d_{GEC}(\epsilon) = O(\log T)$。



整体而言, 这几个框架都是把out-of-sample 的target 转换回到 in-sample 的loss上, 所以我们沿用了 eluder 的名字 ([DMZZ21] 中最早是称为decoupling coefficient)。与这个框架对应的, 另外一个相当惊艳的突破可能是DEC [FKQR21], technical上讲, DEC 和之前一些contextual bandit 上的工作 [FR20, Zha22] 都是将RL问题转换到一个新的 online learning (another out-of-sample target!) 问题上 (这个想法最早可能来自information ratio [RVR14]),  而DEC是这种转换的代价, DEC 在很多问题上有对应的 lower bound, 所以在某种意义上 low DEC 是问题可解的必要条件。不过DEC 的框架也还有一些问题, 一方面它在很多问题上给出的结果的order 会比较差一些, 另一方面它依赖于一些和定义息息相关的, 难以在现实近似的算法设计, 并且没办法分析传统的一些基于优化/采样的OFU-based algorithm。当然最关键的一点, dec 的paper都挺复杂的, 每次一读都得读上一两周, 有机会整理一下我读dec 的note。



## 6 Discussion and Limitation

虽然eluder coefficient 的框架可以 unify 起来 eluder dimension 和 一些 low-rank (bilinear) 类的假设, 并且可以进一步拓展到 POMDP 的情况处理大部分已知的问题 [ZXZ+22], 这个结果其实仍然是比较悲观的结果, 相对于我们现实使用的神经网络而言, 两类结构假设中, [LKFS22] 证明了 Eluder dimension 比generalized linear 模型要更加广一些, 但是似乎没有太好的已知模型; 而另一类的结构假设始终 ``linear in some way''。可能要研究 DRL = DL + RL, 我们首先还是需要再想想 DL 的理论如何发展 :) 



此外, 这类在初始状态上做optimistic modification的算法 $V_{1,f}$ 通常在 $H$-的依赖上都不是最优的。一般而言, 要得到最优的 $H$-依赖, 需要引入方差信息, 这个需要细致的结构条件。一个例子是 [AJZ22]中考虑某种point-wise 的eluder coefficient, 也就是prediction error 和 regularized error 在每个点上的比值, 从而在LSVI算法上获取了更sharp的结果, 但是这类框架通常能处理的问题会更局限一些。





[AOM17] Mohammad Gheshlaghi Azar, Ian Osband, and R ́emi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pages 263–272. PMLR, 2017

[DMZZ21] Christoph Dann, Mehryar Mohri, Tong Zhang, and Julian Zimmert. A provably efficient model-free posterior sampling method for episodic reinforcement learning. Advances in Neural Information Processing Systems, 34:12040–12051, 2021.

[ZZJ20] Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia reference-advantage decomposition. Advances in Neural Information Processing Systems, 33:15198–15207, 2020

[WWK21] Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable mdp with constant suboptimality gap. Advances in Neural Information Processing Systems, 34:9521–9533, 2021.

[JYW21] Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In International Conference on Machine Learning, pages 5084–5096. PMLR, 2021.

[FKQR21] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.

[LLX+23] Zhihan Liu, Miao Lu, Wei Xiong, Han Zhong, Hao Hu, Shenao Zhang, Sirui Zheng, Zhuoran Yang, and Zhaoran Wang. One objective to rule them all: A maximization objective fusing estimation and planning for exploration. arXiv preprint arXiv:2305.18258, 2023.

[ZXZ+22] Han Zhong, Wei Xiong, Sirui Zheng, Liwei Wang, Zhaoran Wang, Zhuoran Yang, and Tong Zhang. GEC: A Unified Framework for Interactive Decision Making in MDP, POMDP, and Beyond. arXiv preprint arXiv:2211.01962, 2022.

[RVR13] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems, 26, 2013

[XFB+22] Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157, 2022.

[AJZ22] Alekh Agarwal, Yujia Jin, and Tong Zhang. Vo q l: Towards optimal regret in model-free rl with nonlinear function approximation. arXiv preprint arXiv:2212.06069, 2022.

[LKFS22] Gene Li, Pritish Kamath, Dylan J Foster, and Nati Srebro. Understanding the eluder dimension. Advances in Neural Information Processing Systems, 35:23737–23750, 2022.

[JLM21] Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. Advances in Neural Information Processing Systems, 34, 2021.

[FR20] Dylan Foster and Alexander Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning, pages 3199–3210. PMLR, 2020.

[Zha22] Tong Zhang. Feel-good thompson sampling for contextual bandits and reinforcement learning. SIAM Journal on Mathematics of Data Science, 4(2):834–857, 2022.

[RVR14] Daniel Russo and Benjamin Van Roy. Learning to optimize via information-directed sampling. Advances in Neural Information Processing Systems, 27, 2014.

[SJK+19] Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In Conference on learning theory, pages 2898–2933. PMLR, 2019.

[JKA+17] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E. Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1704–1713. PMLR, 06–11 Aug 2017.

[DKL+21] Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning, pages 2826–2836. PMLR, 2021.