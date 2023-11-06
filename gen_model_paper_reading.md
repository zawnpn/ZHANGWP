---
share: true
title: Paper Reading
author: 张万鹏 2201111709
institute: 计算机系 数字媒体所
date: '2023/05/24'

mainfont: Times New Roman
CJKmainfont: STSongti-SC-Light
CJKoptions: 
  - BoldFont=STSongti-SC-Bold
  - ItalicFont=STKaiti

header-includes:
  - |
    ````{=latex}
    \usetheme{Madrid}
    \linespread{1.2}
    %\usepackage{some package}
    \usepackage{fontspec}
    \usepackage{cancel}
    \usepackage{ulem}
    \makeatletter
    \newcommand*{\indep}{%
      \mathbin{%
        \mathpalette{\@indep}{}%
      }%
    }
    \newcommand*{\nindep}{%
      \mathbin{%                   % The final symbol is a binary math operator
        %\mathpalette{\@indep}{\not}% \mathpalette helps for the adaptation
        \mathpalette{\@indep}{/}%
                                   % of the symbol to the different math styles.
      }%
    }
    \newcommand*{\@indep}[2]{%
      % #1: math style
      % #2: empty or \not
      \sbox0{$#1\perp\m@th$}%        box 0 contains \perp symbol
      \sbox2{$#1=$}%                 box 2 for the height of =
      \sbox4{$#1\vcenter{}$}%        box 4 for the height of the math axis
      \rlap{\copy0}%                 first \perp
      \dimen@=\dimexpr\ht2-\ht4-.2pt\relax
          % The equals symbol is centered around the math axis.
          % The following equations are used to calculate the
          % right shift of the second \perp:
          % [1] ht(equals) - ht(math_axis) = line_width + 0.5 gap
          % [2] right_shift(second_perp) = line_width + gap
          % The line width is approximated by the default line width of 0.4pt
      \kern\dimen@
      \ifx\\#2\\%
      \else
        \hbox to \wd2{\hss$#1#2\m@th$\hss}%
        \kern-\wd2 %
      \fi
      \kern\dimen@
      \copy0 %                       second \perp
    }
    \makeatother   
    ````
---

![image-20230523235521450](assets/image-20230523235521450.png)

## Outline

- Background & Motivation
- Preliminaries & Problem Formulation
- Method
- Experiments
- Summary

## Background

- Diffusion Model的关键思想是通过去噪过程，将简单的先验分布转化为目标分布，可以将其视作MLE问题。
- 然而Diffusion Model的大多数应用并不直接涉及likelihood，而是具体地应用到downstream task。

----

- 这篇文章主要考虑如何训练Diffusion Model来直接满足这些下游任务目标，而非去匹配某个具体的data distribution。
- 将Diffusion的过程重新定义为一个MDP，然后从MDP的角度使用RL来解决。

----

![image-20230524114045289](assets/image-20230524114045289.png)

## Motivation

- Diffusion Model是一种生成模型，通过模拟随机扩散过程来生成数据。它可以生成一系列的中间状态，并逐渐扩散到最终的数据状态。
- 这个特性使得Diffusion Model非常适合与RL结合，因为RL就是在一系列的状态中选择动作来工作的。

## MDP and RL

Markov Decision Process (MDP) 是决策问题的一种表述，可以定义为$\left({S}, {A}, \rho_0, P, R\right)$。

- 在时刻$t$，agent观测到状态$s_t\in{S}$，执行动作$a_t\in{A}$，接收反馈奖励$R(s_t,a_t)$，转移到$s_{t+1}\sim P(\cdot|s_t,a_t)$。
- agent采取的动作取决于一个policy $\pi(a|s)$。由此可以交互产生序列$\tau=\left(\mathbf{s}_0, \mathbf{a}_0, \mathbf{s}_1, \mathbf{a}_1, \ldots, \mathbf{s}_T, \mathbf{a}_T\right)$。
- RL的objective便是maximize ${J}_{\mathrm{RL}}(\pi)$

$$
{J}_{\mathrm{RL}}(\pi)=\mathbb{E}_{\tau \sim p(\cdot \mid \pi)}\left[\sum_{t=0}^T R\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]
$$

## Problem Formulation

- 假定已有一个diffusion model，其产生的样本分布为$p_\theta\left(\mathbf{x}_0 \mid \mathbf{c}\right)$
- 可以定义denoising diffusion RL objective

$$
{J}_{\mathrm{DDRL}}(\theta)=\mathbb{E}_{\mathbf{c} \sim p(\mathbf{c}), \mathbf{x}_0 \sim p_\theta(\cdot \mid \mathbf{c})}\left[r\left(\mathbf{x}_0, \mathbf{c}\right)\right]
$$

## Denoising as a Multi-step MDP

![image-20230524013310276](assets/image-20230524013310276.png)

- 通过上述定义，可以将${J}_{\mathrm{DDRL}}(\theta)$等价为${J}_\mathrm{RL}(\pi)$
- 这样定义的好处是，通过分解为MDP下每一步的state，可以将denoising procedure下得到的复杂distribution，简化为policy distribution的迭代，进而使用RL进行优化。

----

- 在Diffusion Model中，每一步的噪声添加和去噪过程都可能最终导致复杂的分布。
- 当我们将每一步扩散视为MDP的一步，此时的目标是找到一个policy，使得在每一步选择最优action。
- 在这个框架下，policy通常是一种相对简单的分布，使得我们可以更容易地去估计denoising过程的梯度并进行优化。

## Policy Gradient

Policy Gradient的基本思想是：通过计算policy的梯度，然后沿着policy的方向来更新改进。

对于$J(\theta) = \mathbb{E}[\sum_t \gamma^t R_t |\pi_\theta]$，有如下的结论：
$$
\nabla_\theta J(\theta)=E_\pi\left[\nabla_\theta \log \pi\left(a_t \mid s_t ; \theta\right) G_t\right]
$$
其中，$G_{t} = \sum_{k=t}^T \gamma^{k-t} R_{k}$是从时间$t$开始的累积奖励。最终通过梯度更新$\theta \leftarrow \theta+\alpha \nabla_\theta J(\theta)$便能收敛得到optimal policy。

## Policy Gradient Estimation

为了估计$\nabla_\theta {J}_{\mathrm{DDRL}}$，设计了两种estimator，分别对应强化学习中on-policy和off-policy方法中的梯度：
$$
\hat{g}_{\mathrm{SF}}=\mathbb{E}\left[\sum_{t=0}^T \nabla_\theta \log p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{c}, t, \mathbf{x}_t\right) r\left(\mathbf{x}_0, \mathbf{c}\right)\right]
$$

$$
\hat{g}_{\mathrm{IS}}=\mathbb{E}\left[\sum_{t=0}^T \frac{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{c}, t, \mathbf{x}_t\right)}{p_{\theta_{\text {old }}}\left(\mathbf{x}_{t-1} \mid \mathbf{c}, t, \mathbf{x}_t\right)} \nabla_\theta \log p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{c}, t, \mathbf{x}_t\right) r\left(\mathbf{x}_0, \mathbf{c}\right)\right]
$$

----

- $\hat{g}_{\mathrm{SF}}$是on-policy的，是无偏估计，但只能执行一步更新
- $\hat{g}_{\mathrm{IS}}$是off-policy的，是有偏估计，因此要求更新前后的$p_\theta,p_{\theta_\text{old}}$相差不能太大，但也具有了执行多步更新的能力。
- RL中通过trust region限制更新的幅度来解决上述问题，这篇文章基于PPO算法，提出了 (Denoising Diffusion Policy Optimization) DDPO来解决本篇中想要解决的问题。

## Reward Functions for Text-to-Image Diffusion

为了检验DDPO的表现，这篇文章在text-to-image diffusion上进行验证，因此具体设计了不同的reward function。

- 根据特定的目标来设计reward function，可以直接优化我们关心的目标。
    - 优化生成的图像的美学分数：LAION aesthetics predictor。
    - 优化生成的图像能够被有效地压缩：比较图片压缩前后的文件大小。


----

- **视觉语言模型（VLM）reward**：通过使用一个预训练的VLM自动生成reward。
    - 使用VLM来描述生成的图像，然后将这个描述与原始的prompt进行比较，得到一个相似度reward。![image-20230524120815378](assets/image-20230524120815378.png)

## Experiment

### 主要目标

评估使用RL算法在finetune diffusion model时，对齐各种指定的objective的能力。

----

![image-20230524021102372](assets/image-20230524021102372.png)



----

![image-20230524021205876](assets/image-20230524021205876.png)

----

![image-20230524021409887](assets/image-20230524021409887.png)

## Prompt Alignment

- 实验发现通过DDPO逐渐对齐prompt后，生成的图片变得更加卡通化。
- 作者猜想，由于现实中并不存在这样的图片，因此在pretrain的时候可能使用了卡通化的图片来对应这类prompt。
- 在这样的猜想前提下，进一步说明了DDPO对齐prompt的能力。

## Generalization

![image-20230524125110283](assets/image-20230524125110283.png)

## Overoptimization

![image-20230524125336574](assets/image-20230524125336574.png)

- 如果过度优化reward function，可能会失去原本的语义信息
    - 过度优化压缩性reward，会导致生成的图片几乎都是噪声。
    - 在对齐VLM的生成 ($n$ animals) prompt的reward中，如果过度优化，会导致最终直接写下这个数字，而非生成正确数量的objects。
    - overoptimization也是RL中的一个问题，后续工作可以考虑如何缓解RL带来的这一问题。

## Summary

### Key Idea

将diffusion model的训练重新定义为MDP，并设计具体的reward来引导学习，使其能够满足general goal而非specific distribution。

----

### Thanks for listening!

Q&A
