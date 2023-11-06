---
title: Report
author: Wanpeng Zhang
institute: 
date: 2023/04/07
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
share: true
---

Intuitively:

- causal graph 能够帮助找到$S,A\to S^\prime,R$的因果关系，**并且这个因果关系不会随着non-stationarity带来的变化而变化。**
- 因此基于这样的causal graph学得的policy也不会受non-stationarity影响。

Problem:

- non-stationarity在causal graph的representation?

## Confounder

![image-20230224042732530](report.assets/image-20230224042732530.png){width=85%}

- $(A)$: hard working / $(B)$: performance / $(C)$: intelligent
- 如果直接进行建模$P(B|A)$，随着$P(C)$的改变，会得到错误的model。未考虑样本中$C$对样本分布的影响。
- 考虑$C$进行建模：$P(A,B,C) = P(B|A)P(B|C)P(A|C)P(C)$，得到$B$与$A$之间的正确关系。

----

![image-20230224041711853](report.assets/image-20230224041711853.png)

- 在non-stationary env中，直接学习$P(R|S,A)$不准确。

----

![image-20230224041835971](report.assets/image-20230224041835971.png)

- 将non-stationarity解释为一种影响样本的confounder。
- 通过causal graph的学习，找到了带有confounder的causal structure，本质是因为找到confounder而得到准确的分布：

$$
P(R|O^-_c,A)=\dfrac{P(O,A,R)}{P(R|O_c)P(O^-_c,A|O_c)P(O_c)}
$$

## 2023.03.03

- causal graph帮助解释在给定graph时，加入考虑confounder能够找到正确分布
- 但在non-stationary env中，变化前后的causal graph是否一定保持不变？


----

Consider a family of densities $\{f(\boldsymbol{U} \mid \boldsymbol{W}, \boldsymbol{S}, \boldsymbol{\lambda}): \boldsymbol{\lambda} \in \Lambda\}$ .

![image-20230302202557322](report.assets/image-20230302202557322.png)

- 如果 given distribution family index $\lambda_{{I}_{O_i O_j}}$, $O_i,O_j$ independent，并且$\lambda_{{I}_{O_i}}$和$\lambda_{{I}_{O_j}}$independent，则$\forall O_i,O_j$ independent (given $W,S$).

----

![image-20230302202557322](report.assets/image-20230302202557322.png)

- Thoughts: 将non-stationarity表示为不同分布族 ($f(\cdot|t_i),f(\cdot|t_j)$). If $O_i \nindep O_j|L$, then $\exists O_i\nindep O_j|L,\lambda_{t_it_j|L}$ or $\lambda_{t_i|L}\nindep\lambda_{t_j|L}$.


----

If $O_i \nindep O_j|L$, then $O_i\nindep O_j|L,\lambda_{t_it_j|L}$ $\cancel{\text{or }\lambda_{t_i|L}\nindep\lambda_{t_j|L}}$.

- $\Rightarrow$ If $O_i \nindep O_j|L$, then $\exists O_i\nindep O_j|L,\lambda_{t_it_j|L}$ (+condition/assumption)?
- If $O_i \indep O_j|L$, then $\forall O_i\indep O_j|L,\lambda_{t_it_j|L}$ .

----

- 如果$O_i,O_j$之间原本有一条边（不独立），则$O_i^{t_i},O_j^{t_j}$之间也存在一条边。

- 如果$O_i,O_j$之间原本没有任何边连接（独立），则任意$O_i^{t_i},O_j^{t_j}$之间也不会有边连接。
- 证明了non-stationarity不会改变causal graph的结构。

## 2023.03.10

- 如果$O_i,O_j$之间原本有一条边（不独立），\sout{则$O_i^{t_i},O_j^{t_j}$之间也存在一条边。}则$O_i^{t_i},O_j^{t_j}$之间也**可能**存在一条边。

- 如果$O_i,O_j$之间原本没有任何边连接（独立），则任意$O_i^{t_i},O_j^{t_j}$之间也不会有边连接。
- \sout{证明了non-stationarity不会改变causal graph的结构。}
- stationarity发生变化后的causal graph集合记为$\textbf{G}$, then $\textbf{G}\supseteq G_0$, especially, $\forall G_{new}\in\textbf{G}, N_{edge}(G_{new})\leq N_{edge}(G_0)$.

----

![IMG_0131](report.assets/IMG_0131.jpg)

----

**Intuition**: 我们应该找到$G_{new}=G_0$ (存在这样的$G_{new}$) 使得policy能够adapt non-stationarity.

----

![IMG_0132](report.assets/IMG_0132.jpg)

----

$$
L_{struc}=||(\textbf{G}^\prime-\textbf{G})\odot f(\textbf{G}^\prime-\textbf{G})||_2
$$

$$
f(x)=\alpha\cdot \mathrm{I}(x\geq0)+\beta\cdot \mathrm{I}(x<0), \alpha\geq 0,\beta<0,|\alpha|>|\beta|
$$

- 有向图的邻接矩阵，包含了对direction的惩罚
- penalize "add edge" > penalize "remove edge"

----

![IMG_0135](report.assets/IMG_0135.jpg)

---

$$
L_{sparse} = ||\textbf{G}^\prime||_1
$$

$$
L_{model} = ||\hat{\mathbf{S}}-\mathbf{S}||_2
$$


- $L_{loss} = \lambda_1 L_{model} + \lambda_2 L_{struc} + \lambda_3 L_{sparse}$

- Tune: $\lambda_1, \lambda_2, \lambda_3, \alpha, \beta$

----

### Other thoughts

考虑在model training的结构中如何设计来引导学习confounders

![image-20230224041835971](report.assets/image-20230224041835971.png)

----

![results](report.assets/results.png)

## 2023.03.23

考虑分布族$P(\cdot|\lambda(t))$：表示distribution受non-stationarity影响，在$t$时刻可被观测为$P(\cdot|\lambda(t))$。$\lambda(t)$可以是stochastic的。
$$
P_{\text{mix}}(O_1,O_2|L)=\int_0^T P(O_1,O_2|L,\lambda(t))\cdot p(\lambda(t))\mathrm{d}t
$$

----

- If $O_1 \nindep O_2|L$, then $\exists O_1\nindep O_2|L,\lambda$:

$$
\begin{aligned}
&P_{\text{mix}}(O_1,O_2|L)\neq P_{\text{mix}}(O_1|L)\cdot P_{\text{mix}}(O_2|L)\\
&\exists P(O_1,O_2|L,\lambda(t_1))\neq P(O_1|L,\lambda(t_1))\cdot P(O_2|L,\lambda(t_1))\\
\text{might: }&\exists P(O_1,O_2|L,\lambda(t_2))= P(O_1|L,\lambda(t_2))\cdot P(O_2|L,\lambda(t_2))
\end{aligned}
$$

----

![IMG_0136](report.assets/IMG_0136.jpg)

----


$$
\cancel{L_{struc}=||(\textbf{G}^\prime-\textbf{G})\odot f(\textbf{G}^\prime-\textbf{G})||_2}
$$

- 因为我们不知道underlying truth $\textbf{G}$.

----

![IMG_0131](report.assets/IMG_0131.jpg)

----

- 在更新学习causal graph时，可能部分的边始终保留不变
- 我们应该鼓励在更新过程中保留这些边
- 最大公共子图

----

- 目前没有explicitly learn graph，而是用attention weight来近似表达这种causal information
- $loss = -||\text{Similarity}(W_{new},W_{old})||_\infty$

----

![image-20230324125007590](report.assets/image-20230324125007590.png)

----

next week:

- 考虑explicitly learn edges (but gradients?)
- scheduled restart with MCS?

## 2023.03.31

![image-20230331124021610](report.assets/image-20230331124021610.png)

----

$g\oplus g^\prime$:

- concatenate:  $(g, g_s)$
- element-wise multiplication: $g*g_s$
- weighted-sum: $\alpha\cdot g+(1-\alpha)\cdot g_s$

----

next week:

- implement rest part (change detection, $L_{MCS}$, etc.)
- experiments on dm_control

## 2023.04.07

- $G:(N_\text{nodes}\times N_\text{features})$

$$
\begin{aligned}
S_{G_1,G_2} &= \cos(G_1,G_2)=\frac{G_1 G_2^T}{\lVert G_1 \rVert_2 \cdot\lVert G_2 \rVert_2} \\
G_1^\prime, G_2^\prime &= \text{BipartiteMatching}(S_{G_1,G_2}) \\
L_\text{MCS} &= \lVert S_{G_1,G_2} - S_{G_1^\prime, G_2^\prime} \rVert_2
\end{aligned}
$$

----

-  `if DetectChange(TD-error) == True`: 
    -   `loss += L_mcs`
    -   `unfreeze stable_gnn`
    
-  `else:`
    -  `loss`
    -  `freeze stable_gnn`


----

![image-20230407012231626](report.assets/image-20230407012231626.png)

----

- $g\oplus g_s$:
    - $\lambda_1 g_1+\lambda_2g_2$
    - $(\lambda_1 g_1,\lambda_2g_2)$
    - $\lambda_0s+\lambda_1 g_1+\lambda_2g_2$
    - $(\lambda_0s,\lambda_1 g_1,\lambda_2g_2)$
- env:
    - $s = s+0.1\times \sin(t)$

----

![result](report.assets/result-0842859.png)

----

next week:

- complex envs
- tuning
- +baseline
- `DetectChange(TD-error)`

