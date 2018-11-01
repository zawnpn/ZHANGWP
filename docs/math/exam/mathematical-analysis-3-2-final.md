---
title: 2015-2016第二学期数学分析3-2期末考试（含解答）
date: 2016-07-02 01:38
tags: exam
categories: math
---

> 一、讨论$f(x,y)=\sqrt{|xy|}$在$(0,0)$的可微性.
>
> 二、设$u=u(x)$为方程组
>
$$\left\{\begin{aligned}u=f(x,y,z)\\g(x,y,z)=0\\h(x,y,z)=0\end{aligned}\right.$$
>
确定的隐函数，求$\displaystyle \frac{\mathrm{d}u}{\mathrm{d}x}$.
>
> 三、求$F(x,y,z)=x+y+z$在条件$xy+yz+xz=1$下的条件极值.
>
> 四、求$\displaystyle \iint_{x^2+y^2 \leqslant 1} [y-x] \mathrm{d}x \mathrm{d}y$.(其中$[x]$表示不超过$x$的最大整数)
>
> 五、求$\displaystyle \iint_{|x|+|y| \leqslant 4} \frac{|x^3+xy^2-2x^2-2xy|}{|x|+|y|} \mathrm{d}x \mathrm{d}y$.
>
> 六、求$\displaystyle \iiint_D (x^2+z^2) \mathrm{d}x \mathrm{d}y \mathrm{d}z$.其中区域$D$为由曲面$x^2+y^2=2-z$和$z=\sqrt{x^2+y^2}$所围成的区域.


_感谢Chokie给出了该套题的详细解答，在其授权下将其给出的详解附在本文中。_

<!-- more -->

----------

#### 一、讨论$f(x,y)=\sqrt{|xy|}$在$(0,0)$处的可微性.

> 解：令
>
$$A=\dfrac{\partial f}{\partial x}(0,0)=\displaystyle\lim_{x\rightarrow0}\frac{f(x,0)-f(0,0)}{x}=0$$
>
$$B=\dfrac{\partial f}{\partial y}(0,0)=\displaystyle\lim_{x\rightarrow0}\frac{f(0,y)-f(0,0)}{y}=0$$
>
> 现求极限$\displaystyle\lim_{\substack{x\rightarrow0\\ y\rightarrow0}}\frac{f(x,y)-f(0,0)-Ax-By}{\sqrt{x^{2}+y^{2}}}=\lim_{\substack{x\rightarrow0\\ y\rightarrow0}}\frac{\sqrt{|xy|}}{\sqrt{x^{2}+y^{2}}}$.
>
> 令$x=r\cos\theta,y=r\sin\theta$,
> 则上述极限$=\displaystyle\lim_{r\rightarrow0^{+}}\frac{r\sqrt{|\cos\theta\sin\theta|}}{\sqrt{r^{2}}}=\sqrt{|\cos\theta\sin\theta|}$,
> 可见上述极限显然不存在，则由可微性的定义即有$f(x,y)$在$(0,0)$处不可微.

#### 二、设$u=u(x)$为方程组$\displaystyle \left\{\begin{aligned}u=f(x,y,z)\\g(x,y,z)=0\\h(x,y,z)=0\end{aligned}\right.$确定的隐函数，求$\displaystyle \frac{\mathrm{d}u}{\mathrm{d}x}$.

> 解：对题目中三个等式两端求全微分，得到以下三个等式：
>
>$$\begin{cases}\dfrac{\partial f}{\partial x}\mathrm{d}x+\dfrac{\partial f}{\partial y}\mathrm{d}y+\dfrac{\partial f}{\partial z}\mathrm{d}z=\mathrm{d}u\\ \dfrac{\partial g}{\partial x}\mathrm{d}x+\dfrac{\partial g}{\partial y}\mathrm{d}y+\dfrac{\partial g}{\partial z}\mathrm{d}z=0\\ \dfrac{\partial h}{\partial x}\mathrm{d}x+\dfrac{\partial h}{\partial y}\mathrm{d}y+\dfrac{\partial h}{\partial z}\mathrm{d}z=0\end{cases}$$
>
> 将以上三个等式看作以$\mathrm{d}x,\mathrm{d}y,\mathrm{d}z$为变量的线性方程组，即解得
>
>$\mathrm{d}x=\dfrac{\mathrm{d}u\dfrac{D(g,h)}{D(y,z)}}{\dfrac{D(f,g,h)}{D(x,y,z)}}$,即$\dfrac{\mathrm{d}u}{\mathrm{d}x}=\dfrac{\dfrac{D(f,g,h)}{D(x,y,z)}}{\dfrac{D(g,h)}{D(y,z)}}$.

#### 三、求$F(x,y,z)=x+y+z$在条件$xy+yz+xz=1$下的条件极值.


> 解：令$L(x,y,z)=f(x,y,z)+\lambda(xy+yz+xz-1)(\lambda$待定).
>
> 解方程组
>
$$\begin{cases}\dfrac{\partial L}{\partial x}=1+\lambda(y+z)=0\\ \dfrac{\partial L}{\partial y}=1+\lambda(x+z)=0\\ \dfrac{\partial L}{\partial z}=1+\lambda(x+y)=0\\xy+yz+xz-1=0\end{cases}$$
>
> 即可解得$x=y=z=\dfrac{\sqrt{3}}{3},\lambda=-\dfrac{\sqrt{3}}{2}$或$x=y=z=-\dfrac{\sqrt{3}}{3},\lambda=\dfrac{\sqrt{3}}{2}$.
>
> 而$\mathrm{d}L=(1+\lambda(y+z))\mathrm{d}x+(1+\lambda(x+z))\mathrm{d}y+(1+\lambda(x+y))\mathrm{d}z$,
>$\mathrm{d}^{2}L=\lambda(2\mathrm{d}x\mathrm{d}y+2\mathrm{d}y\mathrm{d}z+2\mathrm{d}x\mathrm{d}z)$.
>
> 由$xy+yz+xz=1$两边取微分得
>
$$(y+z)\mathrm{d}x+(x+z)\mathrm{d}y+(x+y)\mathrm{d}z=0$$
>
由于对上述两解均有$x=y=z$,
> 于是$\mathrm{d}x+\mathrm{d}y+\mathrm{d}z=0$,两边取平方即有
>
$$2\mathrm{d}x\mathrm{d}y+2\mathrm{d}x\mathrm{d}z+2\mathrm{d}y\mathrm{d}z=-(\mathrm{d}^{2}x+\mathrm{d}^{2}y+\mathrm{d}^{2}z)$$
>
> 因此$\mathrm{d}^{2}L=-\lambda(\mathrm{d}^{2}x+\mathrm{d}^{2}y+\mathrm{d}^{2}z)$.
>
> 当$\lambda=-\dfrac{\sqrt{3}}{2}$时,$\mathrm{d}^{2}L>0,f(x,y,z)$取极小值,为$f(\dfrac{\sqrt{3}}{3},\dfrac{\sqrt{3}}{3},\dfrac{\sqrt{3}}{3})=\sqrt{3}.$
>
> 当$\lambda=\dfrac{\sqrt{3}}{2}$时,$\mathrm{d}^{2}L<0,f(x,y,z)$取极大值,为$f(-\dfrac{\sqrt{3}}{3},-\dfrac{\sqrt{3}}{3},-\dfrac{\sqrt{3}}{3})=-\sqrt{3}.$


#### 四、求$\displaystyle \iint_{x^2+y^2 \leqslant 1} [y-x] \mathrm{d}x \mathrm{d}y$.(其中$[x]$表示不超过$x$的最大整数)


> 解：记$D=x^{2}+y^{2}\leqslant1$.易知当$(x,y)\in D,y-x\in[-\sqrt{2},\sqrt{2}]$.
>
> 把区域$D$分成四个区域,
>$D_{1}={(x,y)|-\sqrt{2}\leqslant y-x<-1}\bigcap D$
>$D_{2}={(x,y)|-1\leqslant y-x<0}\bigcap D$
>$D_{3}={(x,y)|0\leqslant y-x<1}\bigcap D$
>$D_{4}={(x,y)|1\leqslant y-x\leqslant\sqrt{2}}\bigcap D$
>
> 则$I=-2\displaystyle\iint_{D_{1}}\mathrm{d}x\mathrm{d}y-\iint_{D_{2}}\mathrm{d}x\mathrm{d}y+\iint_{D_{4}}\mathrm{d}x\mathrm{d}y=-2|D_{1}|-|D_{2}|+|D_{4}|.$
>
> 显然有$|D_{1}|=|D_{4}|$且$|D_{1}|+|D_{2}|=\dfrac{\pi}{2}$,因此$I=-\dfrac{\pi}{2}$.

#### 五、求$\displaystyle \iint_{|x|+|y| \leqslant 4} \frac{|x^3+xy^2-2x^2-2xy|}{|x|+|y|} \mathrm{d}x \mathrm{d}y$.


> 解：记$D=|x|+|y|\leqslant4,|D|=32$,有$|x^{3}+xy^{2}-2x^{2}-2xy|=|x||x^{2}+y^{2}-2x-2y|$.
>
> 区域$D$关于$x,y$是对称的,且根据变量的对称性就有
>
$$I=\displaystyle\iint_{D}\dfrac{|x||x^{2}+y^{2}-2x-2y|}{|x|+|y|}\mathrm{d}x\mathrm{d}y=\iint_{D}\dfrac{|y||x^{2}+y^{2}-2x-2y|}{|x|+|y|}\mathrm{d}x\mathrm{d}y$$
>
> 可得
>
$$I=\displaystyle\dfrac{1}{2}\iint_{D}|x^{2}+y^{2}-2x-2y|\mathrm{d}x\mathrm{d}y=I=\frac{1}{2}\iint_{D}|(x-1)^{2}+(y-1)^{2}-2|\mathrm{d}x\mathrm{d}y$$
>
> 记$D_{1}=(x-1)^{2}+(y-1)^{2}\leqslant2$.由于$D_{1}\subseteq D$,将区域$D$分为两个区域,$D_{1}$和$D\setminus D_{1}$.
>
> 再记
>$I_{1}=\displaystyle\frac{1}{2}\iint_{D_{1}}(2x+2y-x^{2}-y^{2})\mathrm{d}x\mathrm{d}y$
>$I_{2}=\displaystyle\frac{1}{2}\iint_{D\setminus D_{1}}(x^{2}+y^{2}-2x-2y)\mathrm{d}x\mathrm{d}y=\frac{1}{2}\iint_{D}(x^{2}+y^{2}-2x-2y)\mathrm{d}x\mathrm{d}y+I_{1}$
>$I_{3}=\displaystyle\frac{1}{2}\iint_{D}(x^{2}+y^{2}-2x-2y)\mathrm{d}x\mathrm{d}y$
> 则$I=I_{1}+I_{2}=I_{3}+2I_{1}$.
>
> 根据$D_{1}$关于$x,y$的对称性,$I_{1}=\displaystyle\iint_{D_{1}}(1-(x-1)^{2})\mathrm{d}x\mathrm{d}y$,经过极坐标代换容易解出$I_{1}=\pi$.
>
> 同理,有$I_{3}=\displaystyle\iint_{D}((x-1)^{2}-1)\mathrm{d}x\mathrm{d}y=\iint_{D}(x-1)^{2}\mathrm{d}x\mathrm{d}y-32$.
>
> 作变换$u=x+y,v=x-y$,得$x=\dfrac{u+v}{2},y=\dfrac{u-v}{2}$.则$\Big{|}\dfrac{D(x,y)}{D(u,v)}\Big{|}=\dfrac{1}{2}$.
>
> 区域$D\rightarrow D',D'=\{(u,v)|u\in[-4,4],v\in[-4,4]\}$.
>
>$I_{3}=\displaystyle\dfrac{1}{2}\iint_{D'}(\dfrac{u+v}{2}-1)^{2}\mathrm{d}u\mathrm{d}v-32=\frac{1}{2}\int_{-4}^{4}\mathrm{d}v\int_{-4}^{4}(\dfrac{u+v}{2}-1)^{2}\mathrm{d}u-32=\frac{256}{3}$.
>
>$I=\dfrac{256}{3}+2\pi$.

#### 六、求$\displaystyle \iiint_D (x^2+z^2) \mathrm{d}x \mathrm{d}y \mathrm{d}z$.其中区域$D$为由曲面$x^2+y^2=2-z$和$z=\sqrt{x^2+y^2}$所围成的区域.


> 解：联立两曲面方程，消去$z$，即得$D$在$xOy$上的投影为$x^{2}+y^{2}\leqslant1$.
>
> 因此$D=\{(x,y,z)|x^{2}+y^{2}\leqslant1,\sqrt{x^{2}+y^{2}}\leqslant z\leqslant2-x^{2}-y^{2}\}$.
>
>$I=\displaystyle\iint_{x^{2}+y^{2}\leqslant1}\mathrm{d}x\mathrm{d}y\int_{\sqrt{x^{2}+y^{2}}}^{2-x^{2}-y^{2}}(x^{2}+z^{2})\mathrm{d}z$
> 经过坐标变换$x=r\cos\theta,y=r\sin\theta,z=z$,有
>
>$I=\displaystyle\int_{0}^{2\pi}\mathrm{d}\theta\int_{0}^{1}r\mathrm{d}r\int_{r}^{2-r^{2}}(r^{2}\cos^{2}\theta+z^{2})\mathrm{d}z$
>
>$=\displaystyle\int_{0}^{2\pi}cos^{2}\theta\mathrm{d}\theta\int_{0}^{1}r^{3}(2-r^{2}-r)\mathrm{d}r+\frac{2\pi}{3}\int_{0}^{1}r((2-r^{2})^{3}-r^{3})\mathrm{d}r$
>
>$=\displaystyle\pi(\frac{1}{2}-\frac{1}{6}-\frac{1}{5})-\frac{\pi}{3}\int_{0}^{1}(2-r^{2})^{3}\mathrm{d}(2-r^{2})-\frac{2\pi}{15}$
>
>$=-\dfrac{\pi}{12}(2-r^{2})^{4}\Big{|}_{0}^{1}=\dfrac{5\pi}{4}$.
