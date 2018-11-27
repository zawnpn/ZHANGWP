# [读书笔记]Reinforcement Learning: An Introduction - Chapter 2

本章主要针对“非关联性（nonassociative）”的简单场景来学习基础的强化学习方法。什么是“非关联性”呢？其实通过最后 2.9 节可以看出，非关联性在本章就是指**无需考虑每一步行动之间的影响，以及环境对行动的影响**。非关联性问题是一种很理想化的问题，研究这种问题对于现实中的实用性意义不大，但对于入门强化学习理论，是一个不错的背景载体。

再说到强化学习，他与其他的机器学习方法最大的区别，也就是他自身的特点，在于强化学习重点关注**评价性反馈（Evaluative Feedback）**，而不是**指导性反馈（Instructive Feedback）**。

 - 评价性反馈：知道每一步 action 的好坏程度，但不知道这个 action 是否是最好/最差
 - 指导性反馈：直接得知最优 action

指导性反馈多用在监督学习中，需要大量正确的先验知识/信息来给予“指导”，而在一些特殊背景下，无法得到监督性指导，但却有大量实时的评价性反馈，这时候就需要用到**强化学习**。

## 2.1 A k-armed Bandit Problem

### Background

问题的背景就是简化的 k 臂老虎机：

 - 每次在 k 个选项中做出一个选择，称之为一个 action
 - 每次根据玩家的 action 反馈一个“奖励值”，每种 action 对应的奖励值服从一个固定的概率分布（这个概率分布是我们从背后分析问题，也就是从上帝视角才能得知的，真正的玩家一开始根本不知道奖励值服从什么规律或者是否有规律，他需要通过“学习”来找到这一规律）
 - 玩家的目标在于使收获的奖励的**累积值**最大化

### Representation of the problem

$$q_*(a)\doteq\mathbb{E}[R_t|A_t=a]$$

 - $A_t$: 第 t 步做出的 action
 - $R_t$: 第 t 步行动后得到的回报值
 - $a$: 一个任意的行动
 - $q_*(a)$: 行动 a 的理论期望值

我们可以很自然地想到，如果玩家真的从上帝视角得知了这台老虎机的回报规律，也即是知道了每个行动 a 真正能得到理论期望回报$q_*(a)$，那只需保持选择能够收获最大期望值的 action ，就能确保最大的总收益。所以，这个问题的目标，就是要去学习探索，取得关于 $q_*(a), \forall a$ 的信息。

但是玩家一开始显然是不知道 $q_*(a)$ 的情况，所以他要建立一套自己对所有 $a$ 的评估体系，即根据他目前拥有的知识，来估计/猜测当前第 t 步 $a$ 的回报值 $Q_t(a)$: $Q_t(a)\approx q_*(a)$。如何去估计呢？这个先不急，这正是后面长篇大论的东西，简言之，关键在于要有这么一套合适的评估体系。

先假设玩家建立好了一套他认为合适的评估体系，那接下来该如何去根据**评价性反馈**来采取行动呢？这时候先要提到两个概念：

 - 利用（Exploiting）：采取贪心行动，也就是根据目前**已掌握的信息**来做当前最优选择
 - 探索（Exploring）：放弃贪心行动，去探索潜在的、有长远价值的信息

Exploitation 对于每一步而言，是能尽量利用上当前已掌握知识的最佳策略，能确保回报玩家认知范围内的最佳奖励值；Exploration 则会去“试错”，去尝试一些信息量少的 action ，这些 action 之所以信息量少，是因为在玩家的评估体系中被认为是低回报 action 而很少被选中，从而收获到的信息少。不过这个低回报，既有可能是真的低回报，也有可能是被低估了，如果这个 action 事实上是一个很有价值的 action ，却因过分低估而被玩家放弃，是一件非常可惜的事情。为了避免这一情况，从长远意义上真正地最大化收益，就需要玩家适当地去探索、去试错。信息越多，做出的选择也越客观。

## 2.2 Action-value Methods

### Sample-Average

上一节提到，玩家需要建立一套合适的评估体系，这一节就会介绍一种最简单基础的方法。

一个很自然的想法便是将过去得到过的奖励值取均值作为这一次对该 action 的评估

$$Q_{t}(a) \doteq \frac{\sum_{i=1}^{t-1}R_{i} \cdot \textbf{1}_{A_{i} = a}}{\sum_{i=1}^{t-1}\textbf{1}_{A_{i} = a}}$$

我们可以看出：

 - 如果 $\displaystyle \sum_{i=1}^{t-1}\textbf{1}_{A_{i} = a} = 0$, 分母为 0 该式无意义，这时候需要将 $Q_t(a)$ 定义为一个默认值，比如 0
 - 如果 $\displaystyle \sum_{i=1}^{t-1}\textbf{1}_{A_{i} = a}\rightarrow \infty$, 根据**大数定律**, $Q_t(a) \rightarrow q_*(a)$，样本统计值收敛于理论值，达到了我们前面提到的情况——只要掌握了真实的$q_*(a)$，必将能取得最优解。

### greedy action

而前面所提到的贪心行动，表述为数学语言即为

$$A_t \doteq \mathop{\arg\max}\limits_aQ_t(a)$$

可以想象，纯贪心行动很有可能陷入局部最优解（最坏情况下，贪心行动甚至可能导致玩家从头到尾都在选择一个固定的非最优的 action），很难实现让每个 action 都能满足 $\displaystyle \sum_{i=1}^{t-1}\textbf{1}_{A_{i} = a}\rightarrow \infty$ 。

这时候，就需要去“探索（Exploring）”，牺牲一点眼前的利益，换来能带来长远价值的信息。只需对贪心策略稍作修改，我们就能做到这一点。

### ε-greedy action

> **ε-greedy action:** 以 1-ε 的概率采取贪心行动，ε 概率随机选择一个行动 $a$ 。

为什么这个 ε-greedy action 就要比单纯的 greedy action 策略好呢？我们来简单分析一下：

 - 显然可知， ε-greedy action 由于有随机探索的过程，必然能保证：当 $t \rightarrow \infty$，就有 $\displaystyle \sum_{i=1}^{t-1}\textbf{1}_{A_{i} = a}\rightarrow \infty$。这样正如前面已经分析过的，根据大数定律会有 $Q_t(a)\rightarrow q_*(a)$.
 - $\mathrm{Pr}\{A_t=\mathop{\arg\max}\limits_{a}Q_t(a)\} = 1-\varepsilon$，如果 ε 取得太大，就会过于注重探索，而没有充分利用好这些收获到的信息来增加我们的收益，对于我们想要最大化**累积收益**的目标是不利的，但如果取到一个合适的 ε ，便能兼顾信息探索和信息的**充分利用**。

## 2.3 The 10-armed Testbed

这一节就是关于上面提到的方法进行 10-armed bandit 实验来测试效果。

### Background

为了确保实验结果的准确性，总共随机生成了 2000 个 k-armed bandit 问题（k=10），然后针对每个问题，在其背景下都要进行 1000 步 action 的选择，最终针对这 2000 个独立的实验的结果来逐步取均值分析。

<div align="center"><img src="../imgs/RLAI_2/10-armed.png" width="450" alt="10-armed" /></div>

这个图需要好好理解一下，也要根据这张图好好再理解一下问题背景。其中，

 - 首先，需要理解的是，玩家每一步得到的奖励值 $R_t$ ，是一个来自于对应的正态分布的随机值。举个例子，玩家在第 t 步选择 action 3 ，那么这一步老虎机返回给玩家的奖励值 $R_t$ 就是一个服从正态分布 $\mathrm{N}(q_*(3),1)$ 的随机值，这个值或高或低，但总体的趋势还是大概率为 $q_*(3)$ 附近的一个值。我们从上帝视角是知道这些值的，但是玩家并不知道这些情况，只能一步一步地收集信息，以此来猜测、学习这些奖励值的规律。
 - 然后，这些 $q_*(a)$ 是多少呢？这些 $q_*(a)$ 是我们在初始生成 2000 个问题时随机定下来的，我们从标准正态分布 $\mathrm{N}(0,1)$ 中选出 2000 组数据，1 组数据对应生成一个老虎机问题，每组数据有 10 个（所以其实是从 $\mathrm{N}(0,1)$ 中选出了 $2000\times 10=20000$ 个随机数），分别表示这个问题下的 $q_*(1),\ldots,q_*(10)$ 。

### Conclusion

<div align="center"><img src="../imgs/RLAI_2/avg-reward.png" width="450" alt="avg-reward" /></div>

从上图易知，

 - 整体上都能一定程度地通过学习找到问题的规律，所以三种策略最后都能有一个正的稳定回报值（如果没有学到任何信息，也即随机选 action ，最后这些 “**average** reward” 显然会趋于0。注意这个 “**average**” ，是指很多不同问题的平均）
 - 贪心策略一开始的表现要比其他的略好，但是最终明显不如 ε-贪心策略（猜测是陷入了局部最优解）。

<div align="center"><img src="../imgs/RLAI_2/optimal-action.png" width="450" alt="optimal-action" /></div>

从上图易知，

 - 贪心策略只有约 1/3 的次数选到了最优 action，而 ε-贪心策略的表现则显然比单纯的贪心策略好很多，进一步验证了我们认为贪心策略陷入局部最优解的猜想。
 - ε = 0.1 要比 ε = 0.01 选中最优解的概率更大，这与其重视 Exploration 离不开关系，但事实上实验结果表示，最终的总 reward 还是 ε = 0.01 要高一些，原因在于其有 99% 的时间处于 Exploitation 阶段，信息的利用率更高，ε = 0.1 时，Exploitation 的时间只有 90% ，即使探索到了足够多的信息，但是利用率不够高，导致最终效果不如前者。

所以，通过实验，我们看出，Exploration 确实很重要，但是也不能过度探索，需要掌握好平衡。

## 2.4 Incremental Implementation

这一节的简单讲提到如何让计算机来学习 bandit 问题。

### Optimization

首先，我们把问题简化一下，只关注某个具体的 action $a$ ，其他的类比即可。

设 $R_i$ 表示第 i 次选到 $a$ 时系统返回的奖励值，$Q_n$ 表示在前 n 次执行 action $a$ 的经验基础上，对下一次再选到 $a$ 的预测值，那么就有

$$Q_n\doteq \dfrac{R_1+R_2+\cdots + R_{n-1}}{n-1}$$

不难看出，我们一直需要存储每一个 $R_i$ ，空间复杂度为 $O(n)$ ，这显然程序跑到后面，会有着巨大的内存占用。不过，通过一个小技巧便可解决：

$$
\begin{aligned}
Q_{n+1} & = \dfrac{1}{n}\sum_{i=1}^{n}R_i = \dfrac{1}{n}\left(R_n+\sum_{i=1}^{n-1}R_i\right)\\
& = \dfrac{1}{n}\left(R_n+(n-1)\dfrac{1}{n-1}\sum_{i=1}^{n-1}R_i\right)\\
& = \dfrac{1}{n}\left(R_n+(n-1)Q_n\right) = \dfrac{1}{n}\left(R_n+nQ_n-Q_n\right)\\
& = Q_n + \dfrac{1}{n}\left[R_n-Q_n\right],
\end{aligned}
$$

即 $Q_{n+1} = Q_n + \dfrac{1}{n}\left[R_n-Q_n\right]$ ，如此一来，我们只需要存储 $Q, R, n$ ，每次覆写在变量上即可，空间复杂度降为 $O(1)$ ，计算量也有所下降。

上面式子的更广义的写法是

$$NewEstimate \leftarrow OldEstimate + StepSize[Target - OldEstimate]$$

### Pseuducode

在这个基础上，我们整理一下程序的流程，下面是程序的伪代码：

> Initialize, for a = 1 to k:
>
> $\qquad Q(a)\leftarrow 0$
>
> $\qquad N(a)\leftarrow 0$
>
> Repeat forever:
>
> $\qquad A \leftarrow \begin{cases} \mathop{\arg\max}\limits_{a}Q(a) & \text{with probability} \ 1-\epsilon\\ \text{a random action} & \text{with probability} \ \epsilon \end{cases}$
>
> $\qquad R\leftarrow bandit(A)$
>
> $\qquad N(A)\leftarrow N(A)+1$
>
> $\qquad Q(A)\leftarrow Q(A) + \dfrac{1}{N(A)}[R-Q(A)]$

## 2.5 Tracking a Nonstationary Problem

前面的讨论，都是基于**固定奖励值分布**这一条件的，即 $R_t\sim \mathrm{N}(q_*(A_t),1)$ 这一事实在问题生成好之后都是一直保持不变的。然而现实中问题肯定不会如此理想，那么如果这个奖励值分布不是固定不变的，我们该如何解决呢？

显然，这种情况下玩家需要将学习的重心放在每一步近期的奖励值分布情况上，这是因为，奖励值分布的变动，如果有规律的话，无论是周期性还是连续性等，都只会更多地体现在较近时刻，而很久之前的某个 reward 对于这一步而言已经很难看出其影响意义。

一个常见的作法是，将前面提到的增量式中的参数 StepSize 设为一个常量 $\alpha \in (0,1]$ ，则有

$$Q_{n+1}\doteq Q_n + \alpha[R_n-Q_n] (\alpha \in (0,1])$$

（再强调一遍，这几节的很多式子都是为了简化而针对某一个 action 而言的，可以看作是所有 action 的通式，不要理解错了）

我们不断对此式作展开，

$$
\begin{aligned}
Q_{n+1} &\doteq Q_{n} + \alpha[R_{n} - Q_{n}]
\\ &= \alpha R_{n} + (1 - \alpha)Q_{n}
\\ &= \alpha R_{n} + (1 - \alpha)[\alpha R_{n-1} + (1-\alpha)Q_{n-1}]
\\ &= \alpha R_{n} + (1 -\alpha)\alpha R_{n-1} + (1 - \alpha)^{2}Q_{n-1}
\\ &= \alpha R_{n} + (1 - \alpha)\alpha R_{n-1} + (1-\alpha)^{2}\alpha R_{n-2} +...
\\ &+(1-\alpha)^{n-1}\alpha R_{1} + (1-\alpha)^{n}Q_{1}
\\ &= (1-\alpha)^{n}Q_{1}+\sum^{n}_{i=1}(1-\alpha)^{n-i}\alpha R_{i}
\end{aligned}
$$

最终整理得到

$$Q_{n+1} = (1-\alpha)^{n}Q_{1}+\sum^{n}_{i=1}(1-\alpha)^{n-i}\alpha R_{i}$$

因为 $\displaystyle (1-\alpha)^{n}+\sum^{n}_{i=1}\alpha(1-\alpha)^{n-i}=1​$ ，因此这是一个加权平均式，作者将此式称为**指数近因加权平均（Exponential Recency-weighted Average）**。

可以看出，当 i 很大时，$R_i$ 在式子中的影响占比才更大，这也符合了我们要将学习重心放在近期 reward 的要求。下面再讲将 $\alpha$ 设为常量的另一个重要原因。

我们先回到一般情况，对于 $Q_{n+1} = Q_{n} + \alpha_n[R_{n} - Q_{n}]$ ，其中的 $\alpha_n$ 是任意的，也即 step-size 是变长的，对于这样一组 $\{\alpha_n\}$ 序列，如果满足随机逼近理论中的一个条件

$$\sum_{n=1}^{\infty}\alpha_n(a)=\infty\quad \text{and}\quad \sum_{n=1}^{\infty}\alpha_n^2(a)<\infty$$

那么 $Q_n$ 将会以概率 1 收敛。

我们只简单定性分析一下这两个条件的意义：

 - $\displaystyle \sum_{n=1}^{\infty}\alpha_n(a)=\infty$ 能够确保总的步数足够长，进而摆脱初始条件和随机波动的影响。

 - $\displaystyle \sum_{n=1}^{\infty}\alpha_n^2(a)<\infty$ 确保最终的步长足够小，进而能够收敛。

易见，$\alpha_n = \dfrac{1}{n}$ 满足条件能够让其收敛，而 $\alpha_n \equiv \alpha$($\alpha$ 为常量) 则由于不满足条件中的第二项，使 $Q_n$ 不能收敛。看似这是一个坏结果，其实这一结果反而能被利用在**非稳定（nonstationary）问题**中，这是因为，一个不收敛的波动的 $Q_n$ 其实更适合用来描述非稳定问题下的奖励值，而前面收敛的 $Q_n$ 反而可能失去了非稳定环境下的一些关键波动信息。

另一个关键之处在于，满足两个条件的 $\{\alpha_n\}$ 往往收敛缓慢，非常不实用，一般也只用在理论研究中。

## 2.6 Optimistic Initial Values

这一节简单研究了一下初始预估值对模型学习效果的影响。

我们再次拿出前面的指数近因加权平均：

$$Q_{n+1} = (1-\alpha)^{n}Q_{1}+\sum^{n}_{i=1}(1-\alpha)^{n-i}\alpha R_{i}$$

不难看出，前面讨论的所有方法，对每个 action 而言，评估体系显然都会一定程度上受到初始值 $Q_1$ 的影响。在统计学中，这叫做被初值*偏置*了。

初始预估值可以用来根据先验信息提供奖励的期望标准。此外，如果将初始值调高，还有着鼓励模型在早期更多地进行探索的作用。以贪心策略为例，一个很高的初始预期值（称为**乐观初值**），会诱使模型去选择这个 action ，然而事实上 reward 要比估计值差很多，误差值 $[R_n - Q_n]$ 会是一个较大的负数，导致模型对这个 action “失望”，评价降低，下一次，模型就会去主动尝试其他 action 。通过这一方法，达到了鼓励模型在早期多做探索的作用。

下面是一个具体的实验，“高初始值的 greedy 策略” vs “正常初始值的 ε-greedy 策略”。

<div align="center"><img src="../imgs/RLAI_2/optimistic-init.png" width="450" alt="optimistic-init" /></div>

从图片可以看出，即使是纯贪心行动，由于一开始给定较高初始值，模型便如我们分析的一样，在早期进行了大量探索，收获了大量有用的信息，从而也能摆脱局部最优，达到全局最优解，而且其后期几乎 100% 利用率的优势，使其比该实验中 ε-greedy 方法的效果还要优秀。

但是乐观初值法的适用面很窄，它仅适用于固定分布的问题。我们知道模型只会在早期多做探索，后期基本上仍是以 Exploiting 为主，对于非稳定的情况，必然需要时刻探索收集信息，此时乐观初值法就不再适用。

## 2.7 Upper-Confidence-Bound Action Selection

这一节讲到一个考虑得更全面的评估算法：**Upper-Confidence-Bound(UCB)** 算法。简单讲，就是我们之前的 ε-greedy 方法虽然能保证最终能探索到足够的信息，但是效率不高，因为他只是简单的随机探索，探索时每个 action 都是等概率被选择的。

我们可以想一想人是怎样探索学习的。人在探索过程中，通过探索学到的知识，肯定会建立一套标准来判定好坏，如果重复执行某个 action ，一直返回一个低回报，那么必须要动态调整探索策略，适当调低再探索这个 action 的概率，而要尽可能多去探索“潜力”更高的 action 。UCB 算法做的就是这么一件事。

那么 UCB 算法具体是什么呢？

$$A_{t} \doteq \mathop{\arg\max}_{a}\left[Q_{t}(a) + c\sqrt{\frac{\ln{t}}{N_{t}(a)}}\right]$$

UCB 算法就是采取满足上式的 action $A_t$ ，算法的核心就在于我们新加入的 $c\sqrt{\dfrac{\ln t}{N_t(a)}}$ 。

 - $c\sqrt{\dfrac{\ln t}{N_t(a)}}$ ：对于估值的不确定性。更广义地讲，其意义为方差。
 - $c$ ：控制了探索的程度，决定了置信度。
 - $N_t(a)$ ：第 t 步之前 action $a$ 被选中的次数。$N_t(a)$ 如果增加，会给公式中的此项带来降低的影响效果。
 - $\ln t$ ：时间必然会增加，故此项也是一直在保持增加的。但他的影响效果与 action $a$ 的选择状态密切相关。如果 $a$ 不被选择， $\ln t$ 增加但 $N_t(a)$ 不变，故此项变大，不确定性增加；反之，$N_{t+1}(a) = N_t(a) + 1$ ，虽然 $\ln t$ 增加但其增速不如 $N_t(a)$ ，所以此项整体变小，不确定性降低。

所以，通过取 $\mathop{\arg\max}$ 便能动态调整探索策略，适当地提高更加不确定、有潜力的 action 被探索的概率。

<div align="center"><img src="../imgs/RLAI_2/ucb.png" width="450" alt="ucb" /></div>

通过对比实验发现，UCB 算法确实表现要优于普通的 ε-greedy 算法。但是作者也提到，对于更一般的强化学习问题，UCB 算法会遇到一些难点，不再那么适用，比如非稳定问题、大状态空间问题等。

## 2.8 Gradient Bandit Algorithms

### Introduction

前面几种算法，都是在围绕着 $Q_t$ 进行取 $\mathop{\arg\max}$ 然后直接执行 action 的策略，显得有点偏激，一个看上去更合理的做法是，每个 action 对其评分后，确定一个概率分布，然后以这种分布下的**趋势**去做选择，而非凭借数值的绝对大小去做选择。这样显得更加“平滑”，同时根据这些趋势，也能达到动态探索的效果。这便是 Gradient Bandit Algorithms 。

$$\mathrm{Pr}\{A_{t} = a\} \doteq \frac{e^{H_{t}(a)}}{\sum^{k}_{b=1}e^{H_{t}(b)}}\doteq \pi_{t}(a)$$

其中，$H_t(a)$ 是 action $a$ 的偏好值，也就是前面提到的对每个 action 的评分，然后根据 *soft-max* 函数来给出选择每个 action 的概率分布。

而在取得每步的反馈后，我们则需要利用随机梯度上升法来更新偏好值

$$
\begin{aligned}
H_{t+1}(A_{t})&\doteq H_{t}(A_{t}) + \alpha(R_{t}-\overline R_{t})(1-\pi_{t}(A_{t}))
\\H_{t+1}(a) &\doteq H_{t}(a) - \alpha(R_{t} - \overline R_{t})\pi_{t}(a),\ \forall a \neq A_{t}
\end{aligned}
$$

更一般地，我们可以用指示函数来写成一个通式

$$H_{t+1}(a) \doteq H_{t}(a) + \alpha(R_{t}-\overline R_{t})(\textbf{1}_{a = A_t}-\pi_{t}(A_{t}))$$

### Proof

我们知道，随机梯度上升确实能确保收敛到最优值，那么问题就在于，这个形式是否就是“随机梯度上升”的形式呢？

结论先摆出来，上面的方法确实满足随机梯度上升的条件。证明如下：

只需证明

$$
H_{t+1}(a) \doteq H_{t}(a) + \alpha \frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}
$$

其中

$$
\mathbb{E}[R_t] \doteq \sum_{b} \pi_t(b)q_*(b)
$$

而

$$
\begin{aligned}
\frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}&=\frac{\partial}{\partial H_t(a)}\left[\sum_{b}\pi_t(b)q_*(b)\right]
\\&=\sum_b q_*(b)\frac{\partial \pi_t(b)}{\partial H_t(a)}
\end{aligned}
$$

任设一个标量 $X_t$ 与 $b$ 独立，而显然又有 $\displaystyle \sum_b \frac{\partial \pi_t(b)}{\partial H_t(a)} = 0$ ，因此 $\displaystyle X_t\sum_b \frac{\partial \pi_t(b)}{\partial H_t(a)} = 0$ 。接上式，

$$
\begin{aligned}
\\&=\sum_b(q_*(b) - X_t)\frac{\partial \pi_t(b)}{\partial H_t(a)}
\\&=\sum_b \pi_t(b)(q_*(b) - X_t)\frac{\partial \pi_t(b)}{\partial H_t(a)}/\pi_t(b)
\\&=\mathbb{E}[(q_*(A_t) - X_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)]
\end{aligned}
$$

在给定 $A_t$ 的条件下， $\mathbb{E}[R_t|A_t]=q_*(A_t)$ ， $R_t$ 又与其他项不相关，故对于上式，期望意义下可以把 $q_*(A_t)$ 替换为 $R_t$ ，此时再将任设的 $X_t$ 定为 $R_t$ 的均值 $\overline R_t$ ，则有

$$
\frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}=\mathbb{E}[(R_t - \overline R_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)]
$$

由于

$$
\begin{aligned}
\frac{\partial \pi_t(b)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)}\pi_t(b)
\\&= \frac{\partial}{\partial H_t(a)}[\frac{e^{H_{t}(b)}}{\sum^{k}_{c=1}e^{H_{t}(c)}}]
\\&= \frac{\frac{\partial e^{H_t(b)}}{\partial H_t(a)}\sum_{c=1}^k e^{H_t(c)}-e^{H_t(b)} \frac{\partial \sum_{c=1}^k e^{H_t(c)}}{\partial H_t(a)}}{(\sum_{c=1}^k e^{H_t(c)})^2}
\\&=\frac{\textbf{1}_{a = b}e^{H_t(a)}\sum_{c=1}^k e^H_t(c) - e^{H_t(b)}e^{H_t(a)}}{(\sum_{c=1}^k e^{H_t(c)})^2}
\\&=\frac{\textbf{1}_{a = b}e^{H_t(b)}}{\sum_{c=1}^k e^{H_t(c)}} - \frac{e^{H_t(b)}e^{H_t(a)}}{(\sum_{c=1}^k e^{H_t(c)})^2}
\\&=\textbf{1}_{a=b}\pi_t(b)-\pi_t(b)\pi_t(a)
\\&=\pi_t(b)(\textbf{1}_{a=b}-\pi_t(a))
\end{aligned}
$$

代回前面的式子得到

$$
\begin{aligned}
\frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}&=\mathbb{E}[(R_t - \overline R_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)]
\\&=\mathbb{E}[(R_t - \overline R_t)\pi_t(A_t)(\textbf{1}_{a = A_t} - \pi_t(a))/\pi_t(A_t)]
\\&=\mathbb{E}[(R_t - \overline R_t)(\textbf{1}_{a = A_t} - \pi_t(a))]
\end{aligned}
$$

结合上面的结果，此时再来对比我们的两个目标式

$$
\begin{aligned}
H_{t+1}(a) &= H_{t}(a) + \alpha \frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}
\\H_{t+1}(a) &= H_{t}(a) + \alpha(R_{t}-\overline R_{t})(\textbf{1}_{a = A_t}-\pi_{t}(A_{t}))
\end{aligned}
$$

发现确实是梯度上升的形式，证明完毕。

<div align="center"><img src="../imgs/RLAI_2/sga.png" width="450" alt="sga" /></div>

关于 $H_t(a)$ 的更新式中 $\overline R_{t}$ 这一项，他起到一个对比基准线的作用，事实上这个基准线不一定设为均值，他的取值并不影响更新式的方差。作者表明，其实设为均值并不一定能达到最佳效果，但总体而言是一个简单方便且效果较好的一个选择。上图中的实验简单对比了 baseline 为均值和 baseline 为 0 时的不同效果。

## 2.9 Associative Search (Contextual Bandit)

本文的一开头，我们提到本章主要针对“非关联性（nonassociative）”的简单场景来学习基础的强化学习方法。而非关联性在本章就是指**无需考虑每一步行动之间的影响，以及环境对行动的影响**。非关联性问题是一种很理想化的问题，现实中很多东西都是有所联系的，包括 action 与 action 之间的关联， action 与环境之间的关联等等。这一小节，就是关于关联性问题做了一个最基本的简单介绍。

### Background

 - 考虑有 m 个独立的 $k_i$-armed bandit 任务（$i=1,\ldots,m$），每个都有独特的特征能被区分开。
 - 每一步会让你面对一个 $k_i$-armed bandit 任务来做选择。
 - 目标是学习出能将这 m 个独立任务关联起来的最优方案。

### Full Reinforcement Learning Problem

简单而言，之前一直讨论的 nonassociative 问题可以看作现在这个问题下 m=1 的特例。在这个新任务中，我们不但要像之前一样通过探索和利用来学习每个问题的情况，还要把问题之间的关联性也学出来，也就是把环境因素也考虑进来。

这种复杂的问题，叫做 full reinforcement learning problem ，会在书的后面章节讲到。