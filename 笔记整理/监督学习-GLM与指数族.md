# 监督学习：GLM 与指数族

本页统一记号：$m$ 表示样本数，$n$ 表示特征数；$x^{(i)}$ 表示第 $i$ 个样本，$x_j^{(i)}$ 表示第 $i$ 个样本的第 $j$ 列特征；$\hat y$ 表示预测值。

关联：[[监督学习-线性模型]]、[[矩阵基础]]

## 1. 指数族分布

指数族分布写作 $p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$。

其中，$\eta$ 是 natural parameter，自然参数；$T(y)$ 是 sufficient statistic，充分统计量；$b(y)$ 是 base measure，基测度；$a(\eta)$ 是 log-partition function，对数配分函数。

在 CS229 的很多推导里，为了简化，会考虑 $\eta$ 是标量且 $T(y)=y$ 的情况，此时 $p(y;\eta)=b(y)\exp(\eta y-a(\eta))$。

### $a(\eta)$ 的来源

$a(\eta)$ 的作用是让概率分布归一化，也就是 $\int p(y;\eta)dy=1$。

代入指数族形式，得到 $\int b(y)\exp(\eta^TT(y)-a(\eta))dy=1$。

因为 $a(\eta)$ 不依赖 $y$，所以 $e^{-a(\eta)}\int b(y)\exp(\eta^TT(y))dy=1$。

于是有:
		$e^{a(\eta)}=\int b(y)\exp(\eta^TT(y))dy$。

		$a(\eta)=\log\int b(y)\exp(\eta^TT(y))dy$。

记 $Z(\eta)=\int b(y)\exp(\eta^TT(y))dy$，则 $a(\eta)=\log Z(\eta)$，并且 $Z(\eta)=e^{a(\eta)}$。

所以也可以把指数族写成 $p(y;\eta)=\frac{b(y)\exp(\eta^TT(y))}{Z(\eta)}$。

### 一阶导：$a'(\eta)$ 给出均值

先从 $a(\eta)=\log Z(\eta)$ 开始。

对 $\eta$ 求梯度，得到:
		$\nabla_\eta a(\eta)=\frac{1}{Z(\eta)}\nabla_\eta Z(\eta)$。

其中		$Z(\eta)=\int b(y)\exp(\eta^TT(y))dy$。

     $\nabla_\eta Z(\eta)=\int b(y)\exp(\eta^TT(y))T(y)dy$。

代回去，得到 $\nabla_\eta a(\eta)=\frac{\int b(y)\exp(\eta^TT(y))T(y)dy}{Z(\eta)}$。

把分母写回概率分布，得到:
	$\nabla_\eta a(\eta)=\int\frac{ b(y)\exp(\eta^TT(y))T(y)}{Z(\eta)}dy$

	$\nabla_\eta a(\eta)=\int p(y;\eta)T(y)dy$。

因此 $\nabla_\eta a(\eta)=E[T(y)]$。

当 $T(y)=y$ 时，得到 $a'(\eta)=E[y]$。

这就是指数族最重要的性质之一：对数配分函数的一阶导等于充分统计量的期望。

### 二阶导：$a''(\eta)$ 给出方差

下面只看 $\eta$ 是标量、$T(y)=y$ 的情况。

上一节已经得到 
		$a'(\eta)=E[y]$。

继续求导
     $a''(\eta)=\frac{d}{d\eta}E[y]$。

把期望写成积分，$a''(\eta)=\frac{d}{d\eta}\int y\,p(y;\eta)dy$。

把导数移进积分，$a''(\eta)=\int y\frac{\partial p(y;\eta)}{\partial\eta}dy$。

因为
         $\log p(y;\eta)=\log b(y)+\eta y-a(\eta)$。

 故         $\frac{\partial}{\partial\eta}\log p(y;\eta)=y-a'(\eta)$。

         $\frac{\partial p(y;\eta)}{\partial\eta}=p(y;\eta)(y-a'(\eta))$。

代回 $a''(\eta)$，得到 $a''(\eta)=\int y\,p(y;\eta)(y-a'(\eta))dy$
                   $=\int (y-a'(\eta)+a'(\eta))\,p(y;\eta)(y-a'(\eta))dy$

展开为 $a''(\eta)=E[(y-E[y])^2]+a'(\eta)E[y]-a'(\eta)^2\int p(y;\eta)dy$。

因为 $a'(\eta)=E[y]$，$\int p(y;\eta)dy=1$。所以后面抵消 。

因此 $a''(\eta)=Var(y)$。

这说明 $a''(\eta)\ge0$，因为方差永远非负。


## 2. GLM 的基本假设与凸性

GLM 有三个基本假设。

第一，给定 $x$ 后，$y$ 服从某个指数族分布，即 $y\mid x;\theta\sim ExponentialFamily(\eta)$。

第二，模型要预测的是条件均值，即 $h_\theta(x)=E[y\mid x;\theta]$。如果 $T(y)=y$，就是直接预测 $y$ 的均值。

第三，自然参数由输入特征的线性函数给出，即 $\eta=\theta^Tx$。

这个设计的妙处是：$\theta^Tx$ 可以取任意实数，而很多分布的自然参数也正好是无约束的实数。




### GLM 的 NLL Hessian

对单个样本 $(x^{(i)},y^{(i)})$，令 $\eta^{(i)}=\theta^Tx^{(i)}$。

当 $T(y)=y$ 时，log-likelihood 为 $\ell_i(\theta)=\log b(y^{(i)})+y^{(i)}\theta^Tx^{(i)}-a(\theta^Tx^{(i)})$。

负对数似然为 $J_i(\theta)=a(\theta^Tx^{(i)})-y^{(i)}\theta^Tx^{(i)}-\log b(y^{(i)})$。

对 $\theta$ 求梯度，先用链式法则：
	$\nabla_\theta a(\theta^Tx^{(i)})=a'(\eta^{(i)})x^{(i)}$。

	 $\nabla_\theta y^{(i)}\theta^Tx^{(i)}=y^{(i)}x^{(i)}$。

	 $\nabla_\theta J_i(\theta)=(a'(\eta^{(i)})-y^{(i)})x^{(i)}$。

由于 $a'(\eta^{(i)})=E[y\mid x^{(i)};\theta]=h_\theta(x^{(i)})$，因此 $\nabla_\theta J_i(\theta)=(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$。

继续求 Hessian。只有 $a'(\eta^{(i)})$ 依赖 $\theta$。

由链式法则，$\nabla_\theta^2J_i(\theta)=a''(\eta^{(i)})x^{(i)}(x^{(i)})^T$。

又因为 $a''(\eta^{(i)})=Var(y\mid x^{(i)};\theta)\ge0$，所以单样本 Hessian 是 PSD。

对全部样本取平均
	$J(\theta)=\frac1m\sum_{i=1}^mJ_i(\theta)$。

于是 Hessian 为 
	$H=\frac1m\sum_{i=1}^m a''(\eta^{(i)})x^{(i)}(x^{(i)})^T$。

写成矩阵形式，令 $D_{ii}=a''(\eta^{(i)})=Var(y\mid x^{(i)};\theta)$，则 $H=\frac1mX^TDX$。

因为 $D_{ii}\ge0$，所以 $H\succeq0$。因此 GLM 的 NLL 是凸函数。

换成最大似然的角度，log-likelihood 的 Hessian 是 $-H$，所以它是 negative semidefinite。这说明 log-likelihood 是凹函数，最大化时不会有坏的局部最大值。

## 3. 例子：逻辑回归从指数族推出 sigmoid

逻辑回归对应 Bernoulli 分布。设 $y\in\{0,1\}$，$p(y;\phi)=\phi^y(1-\phi)^{1-y}$。

先把它写成指数族形式。

$p(y;\phi)=\exp(y\log\phi+(1-y)\log(1-\phi))$。

展开 $y$ 相关项，得到
$p(y;\phi)=\exp(y\log\frac{\phi}{1-\phi}+\log(1-\phi))$。

根据GLM的假设，有 $\eta=\log\frac{\phi}{1-\phi}$， $T(y)=y$。
可解得 $\phi=\frac{1}{1+e^{-\eta}}$。

又因为 $1-\phi=\frac{1}{1+e^\eta}$，所以 $\log(1-\phi)=-\log(1+e^\eta)$。

所以 $T(y)=y$，$b(y)=1$，$a(\eta)=\log(1+e^\eta)$。

根据假设：
	$\eta=\theta^Tx$。

而预测函数是：
	$h_\theta(x)=E[y\mid x;\theta]=a'(\eta)$。

计算 $a'(\eta)$，有 $a'(\eta)=\frac{e^\eta}{1+e^\eta}=\frac{1}{1+e^{-\eta}}$。

代入 $\eta=\theta^Tx$，得到 $h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$。

这就是 sigmoid 的来源。它不是随便选的函数，而是 Bernoulli 分布在 GLM 假设 $\eta=\theta^Tx$ 下自然推出的 response function。

## 4. 为什么 GLM 有统一的参数更新形式

对于 canonical GLM，单样本 log-likelihood 是 $\ell_i(\theta)=\log b(y^{(i)})+y^{(i)}\theta^Tx^{(i)}-a(\theta^Tx^{(i)})$。

对 $\theta$ 求梯度。

	$\nabla_\theta y^{(i)}\theta^Tx^{(i)}=y^{(i)}x^{(i)}$。

	$\nabla_\theta a(\theta^Tx^{(i)})=a'(\theta^Tx^{(i)})x^{(i)}$。

所以
	$\nabla_\theta \ell_i(\theta)=(y^{(i)}-a'(\theta^Tx^{(i)}))x^{(i)}$。

因为 $a'(\theta^Tx^{(i)})=E[y\mid x^{(i)};\theta]=h_\theta(x^{(i)})$，所以：

	$\nabla_\theta \ell_i(\theta)=(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}$。

因此随机梯度上升更新为：

$\theta:=\theta+\alpha(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}$。

如果最小化 NLL，则方向相反：

$\theta:=\theta-\alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$。

批量形式下，令 $H=h(X)$，则最大化 log-likelihood 的更新是 $\theta:=\theta+\alpha\frac1mX^T(Y-H)$。

这解释了为什么很多线性模型的更新都长得像“真实值减预测值，再乘特征”。

例如：

- 线性回归中，$h_\theta(x)=\theta^Tx$，更新方向是 $(y-h_\theta(x))x$。
- 逻辑回归中，$h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$，更新方向是 $(y-h_\theta(x))x$。
- Poisson 回归中，$h_\theta(x)=e^{\theta^Tx}$，更新方向是 $(y-h_\theta(x))x$。

这些模型的 response function 不同，但在 canonical GLM 框架下，梯度形式统一。
