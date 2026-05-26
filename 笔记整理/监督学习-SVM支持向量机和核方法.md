# 监督学习：SVM 支持向量机和核方法

本页统一记号：$m$ 表示样本数，$n$ 表示特征数；$x^{(i)}$ 表示第 $i$ 个样本，$y^{(i)}$ 表示第 $i$ 个标签。SVM 中默认 $y^{(i)}\in\{-1,1\}$。

关联：[[监督学习-线性模型]]、[[监督学习-GLA生成式学习算法]]、[[数学知识/矩阵基础]]

## 1. 从二分类直觉引入 SVM

在逻辑回归中，我们建模的是条件概率：$p(y=1\mid x)=h_\theta(x)=g(\theta^Tx)$，其中 $g(z)=\frac{1}{1+e^{-z}}$。

当 $h_\theta(x)\ge 0.5$ 时，模型预测为正类。因为 sigmoid 在 $z=0$ 时等于 $0.5$，所以 $h_\theta(x)\ge 0.5$ 等价于 $\theta^Tx\ge 0$。

从直觉上看：

- 对于正样本 $y=1$，我们希望 $\theta^Tx$ 越大越好。
- 对于负样本 $y=0$，我们希望 $\theta^Tx$ 越小越好。

因为 $\theta^Tx$ 越大，经过 sigmoid 后 $p(y=1\mid x)$ 越接近 $1$；$\theta^Tx$ 越小，$p(y=1\mid x)$ 越接近 $0$。

SVM 的出发点稍微不同。它不直接建模概率，而是希望找到一条能够把两类样本分开的分类边界，并且希望这条边界两侧留出尽可能大的间隔。

也就是说，SVM 的核心思想是：不只是把样本分开，而是要分得尽可能清楚。

## 2. SVM Notation

在 SVM 中，通常把二分类标签写成 $y\in\{-1,1\}$。

分类器写成 $h_{w,b}(x)=g(w^Tx+b)$，其中：

$g(z)=1$，若 $z\ge 0$。

$g(z)=-1$，若 $z<0$。

分类边界是 $w^Tx+b=0$。

如果 $w^Tx+b\ge 0$，则预测为 $1$；否则预测为 $-1$。

对于训练样本 $(x^{(i)},y^{(i)})$，可以用 $y^{(i)}(w^Tx^{(i)}+b)$ 统一表示分类是否正确。

如果 $y^{(i)}(w^Tx^{(i)}+b)>0$，说明分类正确。

原因是：

- 若 $y^{(i)}=1$，分类正确要求 $w^Tx^{(i)}+b>0$。
- 若 $y^{(i)}=-1$，分类正确要求 $w^Tx^{(i)}+b<0$。

两种情况合起来就是 $y^{(i)}(w^Tx^{(i)}+b)>0$。

## 3. Functional Margin

定义单个样本的函数间隔 functional margin：

$\hat\gamma^{(i)}=y^{(i)}(w^Tx^{(i)}+b)$。

如果 $\hat\gamma^{(i)}>0$，说明样本被正确分类。

如果 $\hat\gamma^{(i)}$ 很大，说明不仅分类正确，而且在代数意义上离分类边界更远。

整个训练集上的函数间隔定义为所有样本函数间隔的最小值：

$\hat\gamma=\min_i\hat\gamma^{(i)}=\min_i y^{(i)}(w^Tx^{(i)}+b)$。

SVM 希望最大化最小间隔，也就是让最靠近边界的点也尽可能远离边界。
![[SVMmargin.png|的核心假设]]
## 4. Functional Margin 的缺陷

函数间隔有一个问题：它会受到参数缩放的影响。

如果把 $w,b$ 同时乘以一个正数 $c$，即 $w'=cw$，$b'=cb$，那么分类边界变成 $w'^Tx+b'=c(w^Tx+b)=0$。

这和原来的边界 $w^Tx+b=0$ 表示同一条分类边界。

但是函数间隔会变成：

$y^{(i)}(w'^Tx^{(i)}+b')=c\,y^{(i)}(w^Tx^{(i)}+b)$。

也就是说，分类边界没有变，但 functional margin 被放大了 $c$ 倍。

因此，functional margin 不能真正表示样本到分类边界的几何距离。

这也能解释为什么 linearly separable logistic regression 会出现参数不断变大的现象：如果只看代数分数，继续放大参数仍然可以让分数更极端，但边界本身并没有变。

## 5. Geometric Margin

为了消除缩放影响，引入几何间隔 geometric margin。
![[SVM几何间隔.png|的核心假设]]

点 $x$ 到超平面 $w^Tx+b=0$ 的距离是 $\frac{|w^Tx+b|}{\|w\|_2}$。

可以这样理解：从点 $x$ 沿着法向量 $w$ 的方向走到超平面。设最近点为 $x+t w$，它在超平面上，所以 $w^T(x+tw)+b=0$。

展开得到 $w^Tx+b+t\|w\|_2^2=0$，所以 $t=-\frac{w^Tx+b}{\|w\|_2^2}$。

两点距离是 $\|tw\|_2=|t|\|w\|_2=\frac{|w^Tx+b|}{\|w\|_2}$。

对于带标签样本，定义带符号的几何间隔：

$\gamma^{(i)}=\frac{y^{(i)}(w^Tx^{(i)}+b)}{\|w\|_2}$。

因此，函数间隔和几何间隔的关系是：

$\gamma^{(i)}=\frac{\hat\gamma^{(i)}}{\|w\|_2}$。

整个训练集的几何间隔为：

$\gamma=\min_i\gamma^{(i)}=\min_i\frac{y^{(i)}(w^Tx^{(i)}+b)}{\|w\|_2}$。

几何间隔不会因为同时缩放 $w,b$ 而改变，所以它才是真正有意义的距离。

## 6. 从最大几何间隔到硬间隔 SVM

如果训练集线性可分，SVM 希望找到一个分类超平面，使训练集的最小几何间隔最大。

原始目标可以写成：最大化 $\gamma$，并满足 $\frac{y^{(i)}(w^Tx^{(i)}+b)}{\|w\|_2}\ge \gamma$，对所有 $i=1,\dots,m$ 成立。

这个形式不方便直接优化。由于同一条分类边界可以由无数个成比例的 $(w,b)$ 表示，所以可以人为规定最小 functional margin 为 $1$。

也就是规定：

$y^{(i)}(w^Tx^{(i)}+b)\ge 1$。

此时最近点的 functional margin 为 $1$，几何间隔为 $\gamma=\frac{1}{\|w\|_2}$。

最大化几何间隔等价于最小化 $\|w\|_2$，也等价于最小化 $\frac12\|w\|_2^2$。

因此硬间隔 SVM 的 primal problem 是：

最小化 $\frac12\|w\|_2^2$。

约束为 $y^{(i)}(w^Tx^{(i)}+b)\ge 1$，对所有 $i=1,\dots,m$ 成立。

这里的 $\frac12$ 只是为了求导方便，因为 $\nabla_w\frac12\|w\|_2^2=w$。

## 7. 支持向量

在硬间隔 SVM 中，满足 $y^{(i)}(w^Tx^{(i)}+b)=1$ 的点，正好位于 margin 边界上。

这些点被称为支持向量 support vectors。

它们是离分类边界最近的训练样本，也是决定最终分类超平面的关键点。

如果某个点离边界很远，那么它对最终边界的位置影响通常不大；真正卡住边界的是支持向量。

## 8. 软间隔 SVM

![[SVM软间隔.png|Pasted image 20260526011851.png]]

硬间隔 SVM 要求所有样本都被完全正确分类，而且都满足 $y^{(i)}(w^Tx^{(i)}+b)\ge 1$。

但现实数据往往不可线性分开，或者存在噪声、异常点。

因此引入松弛变量 $\xi_i\ge 0$，允许某些样本违反 margin 约束：

$y^{(i)}(w^Tx^{(i)}+b)\ge 1-\xi_i$。

软间隔 SVM 的优化问题为：

最小化 $\frac12\|w\|_2^2+C\sum_{i=1}^m\xi_i$。

约束为 $y^{(i)}(w^Tx^{(i)}+b)\ge 1-\xi_i$，且 $\xi_i\ge 0$。

其中 $C$ 控制两个目标之间的权衡：

- 让 margin 尽可能大。
- 让违反 margin 的程度尽可能小。

当 $C$ 较大时，模型更重视训练集上的分类正确性，容错更低，可能更容易过拟合。

当 $C$ 较小时，模型更允许一些样本违反 margin，边界可能更平滑，泛化可能更好。

从 loss 的角度，软间隔 SVM 对应 hinge loss：

$\ell_i(w,b)=\max(0,1-y^{(i)}(w^Tx^{(i)}+b))$。

如果 $y^{(i)}(w^Tx^{(i)}+b)\ge 1$，则 hinge loss 为 $0$。

如果 $y^{(i)}(w^Tx^{(i)}+b)<1$，说明样本要么分错，要么虽然分对但 margin 不够大，因此产生损失。

这和 logistic loss 不同。Logistic loss 永远不会严格等于 $0$，但 hinge loss 一旦 margin 达到 $1$ 就不再继续奖励更大的 functional margin。

## 9. 对偶问题与 $w$ 的线性组合表示

SVM 的原问题是一个带约束的优化问题。对于硬间隔 SVM：

最小化 $\frac12\|w\|_2^2$。

约束为 $y^{(i)}(w^Tx^{(i)}+b)\ge 1$。

可以把约束写成 $1-y^{(i)}(w^Tx^{(i)}+b)\le 0$。

构造拉格朗日函数：

$L(w,b,\alpha)=\frac12\|w\|_2^2+\sum_{i=1}^m\alpha_i[1-y^{(i)}(w^Tx^{(i)}+b)]$。

其中 $\alpha_i\ge 0$。

对 $w$ 求导并令其为 $0$：

$\nabla_w L=w-\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}=0$。

所以：

$w=\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}$。

它说明：最优的 $w$ 可以写成训练样本的线性组合。

注意 $y^{(i)}\in\{-1,1\}$ 只是标量，因此 $w=\sum_i\alpha_i y^{(i)}x^{(i)}$ 仍然是训练样本 $x^{(i)}$ 的线性组合。

对 $b$ 求导还可以得到：

$\sum_{i=1}^m\alpha_i y^{(i)}=0$。

最终的硬间隔 SVM 对偶问题是：

最大化 $\sum_{i=1}^m\alpha_i-\frac12\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy^{(i)}y^{(j)}\langle x^{(i)},x^{(j)}\rangle$。

约束为 $\alpha_i\ge 0$，且 $\sum_{i=1}^m\alpha_i y^{(i)}=0$。

对于软间隔 SVM，对偶问题的主要变化是 $0\le\alpha_i\le C$。

## 10. 为什么对偶形式重要

预测函数原本是 $f(x)=w^Tx+b$。

由 $w=\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}$ 可得：

$f(x)=(\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)})^Tx+b$。

展开：

$f(x)=\sum_{i=1}^m\alpha_i y^{(i)}\langle x^{(i)},x\rangle+b$。

这说明 SVM 的预测只依赖于训练样本和新样本之间的内积。

而且很多 $\alpha_i$ 会等于 $0$，真正留下来的通常是支持向量。

因此实际预测可以写成：

$f(x)=\sum_{i\in SV}\alpha_i y^{(i)}\langle x^{(i)},x\rangle+b$。

其中 $SV$ 表示支持向量集合。

## 11. 表示定理的两个直觉

### 11.1 从梯度更新角度理解

以线性回归或逻辑回归的梯度下降为例，参数更新经常具有如下形式：

$\theta:=\theta+\alpha\sum_{i=1}^m c_i x^{(i)}$。

其中 $c_i$ 是某个标量系数，可能包含误差项、标签、学习率等。

如果初始化 $\theta=0$，那么第一次更新后，$\theta$ 仍然是训练样本的线性组合。

之后每一次更新都是“原来的线性组合”再加上“一组训练样本的线性组合”，所以 $\theta$ 始终留在训练样本张成的空间中。

如果使用 feature map $x\mapsto \phi(x)$，那么同理：

$\theta:=\theta+\alpha\sum_{i=1}^m c_i\phi(x^{(i)})$。

若初始化为 $0$，则始终有：

$\theta=\sum_{i=1}^m\beta_i\phi(x^{(i)})$。

### 11.2 从平行/垂直分解角度理解

设训练样本张成的空间为 $\mathrm{span}\{x^{(1)},\dots,x^{(m)}\}$。

任意一个 $w$ 都可以分解成 $w=w_\parallel+w_\perp$。

其中：

- $w_\parallel$ 在训练样本张成的空间中。
- $w_\perp$ 与所有训练样本正交。

因此对任意训练样本 $x^{(i)}$，有 $w_\perp^Tx^{(i)}=0$。

于是：

$w^Tx^{(i)}=w_\parallel^Tx^{(i)}+w_\perp^Tx^{(i)}=w_\parallel^Tx^{(i)}$。

也就是说，$w_\perp$ 不影响训练样本上的预测值，也不影响训练约束。

但范数会变大：

$\|w\|_2^2=\|w_\parallel\|_2^2+\|w_\perp\|_2^2$。

在 SVM 中，我们要最小化 $\frac12\|w\|_2^2$，所以 $w_\perp$ 只会增大目标函数，却不会改善训练样本上的分类约束。

因此最优解不会保留 $w_\perp$，也就是说 $w=w_\parallel$。

所以最优的 $w$ 位于训练样本张成的空间中，可以写成训练样本的线性组合。

注意：这里说的是 $w_\perp$ 不影响训练样本上的函数值和约束，不是说它完全不影响整个输入空间中所有点的分类边界。

## 12. Feature Map：在线性模型中引入非线性

![[SVM划分.png|的核心假设]]

原来的线性模型是 $w^Tx+b$。

如果原始输入空间中数据不能被线性分开，可以先做特征映射：

$\phi:\mathbb R^n\to\mathbb R^p$。

将原始输入 $x$ 映射成更高维的特征向量 $\phi(x)$。

模型变成 $w^T\phi(x)+b$。

虽然它对 $\phi(x)$ 是线性的，但对原始输入 $x$ 来说可以是非线性的。

例如原始输入 $x=[x_1,x_2]^T$，定义 $\phi(x)=[x_1^2,\ x_1x_2,\ x_2^2]^T$。

则模型为 $w^T\phi(x)=w_1x_1^2+w_2x_1x_2+w_3x_2^2$。

这对原始输入 $x$ 来说是一个二次函数，不再是线性的。

所以 feature map 的核心是：把原始输入变成更丰富的特征表示，然后在新特征空间里做线性学习。

## 13. Kernel Trick：不显式计算高维特征

如果显式构造 $\phi(x)$，维度可能非常高。

例如对于 $x\in\mathbb R^n$，如果构造所有不超过 3 次的多项式特征，那么特征维度大约是 $O(n^3)$。

当 $n=1000$ 时，维度大约是 $10^9$，这几乎无法显式计算和存储。

核方法的关键是：很多算法最终只依赖特征向量之间的内积 $\langle \phi(x),\phi(z)\rangle$。

因此定义核函数：

$K(x,z)=\phi(x)^T\phi(z)$。

核函数的作用不是直接把 $x$ 变成高维向量，而是直接计算映射到高维空间之后，两个特征向量的内积。

如果一个算法只依赖内积，就可以把 $\langle \phi(x),\phi(z)\rangle$ 替换成 $K(x,z)$，从而避免显式构造 $\phi(x)$。

## 14. 核 SVM 的预测形式

在特征空间中，SVM 的分类函数为：

$f(x)=w^T\phi(x)+b$。

由表示形式 $w=\sum_{i=1}^m\alpha_i y^{(i)}\phi(x^{(i)})$ 代入可得：

$f(x)=(\sum_{i=1}^m\alpha_i y^{(i)}\phi(x^{(i)}))^T\phi(x)+b$。

展开：

$f(x)=\sum_{i=1}^m\alpha_i y^{(i)}\phi(x^{(i)})^T\phi(x)+b$。

使用核函数 $K(x^{(i)},x)=\phi(x^{(i)})^T\phi(x)$，得到：

$f(x)=\sum_{i=1}^m\alpha_i y^{(i)}K(x^{(i)},x)+b$。

如果只考虑支持向量，则：

$f(x)=\sum_{i\in SV}\alpha_i y^{(i)}K(x^{(i)},x)+b$。

分类结果为 $\hat y=\mathrm{sign}(f(x))$。

这就是核 SVM 的预测形式。

## 15. 核矩阵与 Mercer 条件

给定训练集 $x^{(1)},x^{(2)},\dots,x^{(m)}$，定义核矩阵：

$K_{ij}=K(x^{(i)},x^{(j)})$。

如果 $K$ 是合法核函数，即存在某个特征映射 $\phi$，使得 $K(x,z)=\phi(x)^T\phi(z)$，那么核矩阵必须满足两个性质。

### 15.1 对称性

因为内积对称：

$\phi(x^{(i)})^T\phi(x^{(j)})=\phi(x^{(j)})^T\phi(x^{(i)})$。

所以 $K_{ij}=K_{ji}$，即核矩阵是对称矩阵。

### 15.2 半正定性

对于任意向量 $a\in\mathbb R^m$，有 $a^TKa\ge 0$。

证明如下：

$a^TKa=\sum_{i=1}^m\sum_{j=1}^m a_i a_j K(x^{(i)},x^{(j)})$。

代入核函数定义：

$a^TKa=\sum_{i=1}^m\sum_{j=1}^m a_i a_j\phi(x^{(i)})^T\phi(x^{(j)})$。

整理为：

$a^TKa=\left\|\sum_{i=1}^m a_i\phi(x^{(i)})\right\|_2^2\ge 0$。

因此核矩阵是半正定的。

Mercer 定理可以粗略理解为：一个函数 $K(x,z)$ 是合法核函数，当且仅当对于任意有限样本集合，它生成的核矩阵都是对称半正定的。

注意：不是只要在某一个数据集上核矩阵半正定就一定够，而是要求对任意有限样本集合都成立。

## 16. 补充：LMS With Kernel Trick

这一节不是 SVM 本身，而是帮助理解 kernel trick 的一个例子。

考虑使用 feature map 后的 LMS 模型：

$h_\theta(x)=\theta^T\phi(x)$。

Batch gradient descent 更新为：

$\theta:=\theta+\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})$。

假设初始化 $\theta=0$。

由于每次更新都是若干个 $\phi(x^{(i)})$ 的线性组合，所以始终有：

$\theta=\sum_{i=1}^m\beta_i\phi(x^{(i)})$。

代入预测项：

$\theta^T\phi(x^{(i)})=(\sum_{j=1}^m\beta_j\phi(x^{(j)}))^T\phi(x^{(i)})$。

得到：

$\theta^T\phi(x^{(i)})=\sum_{j=1}^m\beta_j\phi(x^{(j)})^T\phi(x^{(i)})$。

使用核函数 $K(x^{(j)},x^{(i)})=\phi(x^{(j)})^T\phi(x^{(i)})$：

$\theta^T\phi(x^{(i)})=\sum_{j=1}^m\beta_jK(x^{(j)},x^{(i)})$。
所以预测时根本不需要知晓$\phi$，只用知道核矩阵核$\beta$即可

$\theta:=\theta+\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})$
$\theta:=\sum_{i=1}^m\beta_i\phi(x^{(i)})+\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})$
  $=\sum_{i=1}^m(\beta_i+\alpha(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})$。

所以 $\beta_i$ 的更新可以写成：

$\beta_i:=\beta_i+\alpha(y^{(i)}-\theta^T\phi(x^{(i)})=\beta_i+\alpha(y^{(i)}-\sum_{j=1}^m\beta_jK(x^{(j)},x^{(i)}))$。

如果定义核矩阵 $K_{ij}=K(x^{(i)},x^{(j)})$，则向量形式为：

$\beta:=\beta+\alpha(y-K\beta)$。

预测时：

$\theta^T\phi(x)=(\sum_{i=1}^m\beta_i\phi(x^{(i)}))^T\phi(x)$。

所以：

$\theta^T\phi(x)=\sum_{i=1}^m\beta_iK(x^{(i)},x)$。

这说明：训练和预测都只需要核函数值，不需要显式知道 $\phi(x)$，也不需要显式存储高维参数 $\theta$。

代码示例:

注意，下面采用的时SGD更新，每次只用一个样本更新，相当于每次只更新$\beta_i$
这里的$h_\theta(x)=g(\theta^T\phi(x))$所以先算里面再套个g

```python
def update_state(state, kernel, learning_rate, x_i, y_i):

    """Updates the state of the perceptron.
    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    z=0
    
    for beta,x in zip(state["beta"],state["X"]):
		
        z+=beta*kernel(x,x_i)

    g=sign(z)
    beta_i=learning_rate*(y_i-g)
    state["beta"]+=[beta_i]
    state["X"]+=[x_i]

    return state
```
![[SVM高斯核结果.png]]
## 17. 常见核函数

### 17.1 线性核

线性核是 $K(x,z)=x^Tz$。

它对应的 feature map 是 $\phi(x)=x$。

使用线性核等价于在原始空间中学习线性模型。

### 17.2 多项式核

假设 $\phi(x)$ 包含所有 0 到 3 次的多项式特征，并且为了简单起见，包含重复项。

例如：

$\phi(x)=[1,\ x_1,\dots,x_n,\ x_1x_1,\ x_1x_2,\dots,\ x_ix_jx_k,\dots]^T$。

那么：

$\langle \phi(x),\phi(z)\rangle=1+\sum_i x_i z_i+\sum_{i,j}x_ix_jz_iz_j+\sum_{i,j,k}x_ix_jx_kz_iz_jz_k$。

注意 $\sum_i x_i z_i=\langle x,z\rangle$。

而 $\sum_{i,j}x_ix_jz_iz_j=(\sum_i x_i z_i)^2=\langle x,z\rangle^2$。

同理，$\sum_{i,j,k}x_ix_jx_kz_iz_jz_k=(\sum_i x_i z_i)^3=\langle x,z\rangle^3$。

所以：

$\langle \phi(x),\phi(z)\rangle=1+\langle x,z\rangle+\langle x,z\rangle^2+\langle x,z\rangle^3$。

因此可以定义核函数：

$K(x,z)=1+\langle x,z\rangle+\langle x,z\rangle^2+\langle x,z\rangle^3$。

这样就不用显式构造 $O(n^3)$ 维的 $\phi(x)$。

更常见的多项式核写成：

$K(x,z)=(x^Tz+c)^d$。

### 17.3 高斯核 / RBF 核

高斯核，也叫 RBF 核：

$K(x,z)=\exp(-\frac{\|x-z\|_2^2}{2\sigma^2})$。

如果 $x$ 和 $z$ 很接近，则 $K(x,z)\approx 1$。

如果 $x$ 和 $z$ 很远，则 $K(x,z)\approx 0$。

因此高斯核可以理解为一种基于距离的相似度函数。

参数 $\sigma$ 控制影响范围：

- $\sigma$ 较小：每个样本只影响附近很小区域，边界可能更复杂。
- $\sigma$ 较大：影响范围更宽，边界更平滑。

## 18. 总结主线

SVM 和核方法的整体逻辑可以概括为：

1. 二分类中，我们希望不仅分对样本，还希望分类边界足够清晰。
2. SVM 用 margin 描述样本到分类边界的间隔。
3. Functional margin 会受到参数缩放影响，因此引入 geometric margin。
4. 最大化 geometric margin 可以转化为最小化 $\frac12\|w\|_2^2$，并加入约束 $y^{(i)}(w^Tx^{(i)}+b)\ge 1$。
5. 对偶推导得到 $w=\sum_i\alpha_i y^{(i)}x^{(i)}$，所以预测只依赖样本之间的内积。
6. 如果使用 feature map $x\mapsto\phi(x)$，则模型可以在高维空间中线性分类，从而在原始空间表现为非线性边界。
7. 因为算法只依赖 $\phi(x)^T\phi(z)$，所以可以用核函数 $K(x,z)=\phi(x)^T\phi(z)$ 来避免显式构造高维特征。
8. 合法核函数对应的核矩阵应当是对称半正定的。
9. 常见核函数包括线性核、多项式核和高斯核。



