# ICA 独立成分分析

关联：[[无监督学习-PCA主成分分析]]、[[数学知识/概率论基础]]、[[数学知识/矩阵基础]]

ICA（Independent Component Analysis，独立成分分析）是一种用于盲源分离的无监督学习方法。它的目标是：从多个观测到的混合信号中，恢复出背后相互独立的原始源信号。

ICA 和 PCA 都是在寻找一组新的表示基，但二者目标不同。PCA 寻找最大方差方向，使投影后的特征不相关；ICA 寻找独立源方向，使恢复出的成分尽可能统计独立。因此，PCA 更关注协方差结构，而 ICA 更关注概率分布中的独立性与非高斯性。

## 1. 鸡尾酒会问题

CS229 讲义中用“鸡尾酒会问题”引入 ICA：多个说话人在房间里同时说话，不同位置的麦克风录到的是这些声音的不同线性混合。ICA 的目标就是仅凭这些混合录音，分离出原始说话人的声音。
![[ICAjpg.jpg|Pasted image 20260527234513.png|697]]
假设房间里有两个说话人，原始声音信号分别为 $s_1,s_2$；房间里有两个麦克风，录到的混合信号为 $x_1,x_2$。每个麦克风录到的都不是某一个人的纯声音，而是两个说话人声音的线性组合：

$$
x_1=a_{11}s_1+a_{12}s_2
$$

$$
x_2=a_{21}s_1+a_{22}s_2
$$

我们只知道观测信号 $x_1,x_2$，但不知道原始源信号 $s_1,s_2$，也不知道混合系数 $a_{ij}$。ICA 要解决的问题就是：在 $s$ 和 $A$ 都未知的情况下，仅凭观测信号 $x$，恢复出原始独立源信号 $s$。

这类问题也叫盲源分离（Blind Source Separation, BSS）。其中“盲”的意思是：我们不知道混合矩阵，也不知道原始源信号。

## 2. 数学建模

设有 $d$ 个独立源信号 $s\in\mathbb R^d$，观测到的混合信号为 $x\in\mathbb R^d$。ICA 假设观测信号是源信号经过未知线性混合矩阵 $A$ 得到的：

$$
x=As
$$

其中 $A\in\mathbb R^{d\times d}$ 称为混合矩阵。基础 ICA 模型假设 $A$ 可逆，于是可以定义解混矩阵 $W=A^{-1}$，并得到 $s=Wx$。

对于第 $i$ 个样本，有 $s^{(i)}=Wx^{(i)}$。讲义中将 $W$ 的第 $j$ 行记为 $w_j^T$，因此恢复出的第 $j$ 个源信号为：

$$
s_j=w_j^Tx
$$

这句话很重要。ICA 本质上就是在寻找若干个向量 $w_1,w_2,\dots,w_d$，使得 $w_1^Tx,w_2^Tx,\dots,w_d^Tx$ 尽可能像彼此独立的源信号。

## 3. ICA 的核心假设

ICA 能够工作，依赖几个关键假设。

第一，源信号的各个分量彼此统计独立，也就是 $p(s)=\prod_{j=1}^d p_j(s_j)$。这里的独立不是简单的“不相关”，而是更强的概率意义上的独立。PCA 通常只要求投影后的分量不相关，即 $\operatorname{Cov}(y_i,y_j)=0$；ICA 要求恢复出的源成分尽可能独立。

第二，源信号通常要求是非高斯的。更准确地说，在标准 ICA 模型中，最多只能有一个源成分服从高斯分布。原因是高斯分布具有旋转不变性，如果 $s\sim\mathcal N(0,I)$，那么对任意正交矩阵 $R$，$Rs$ 和 $s$ 具有相同分布。因此，当源信号是多维高斯时，我们无法仅从观测数据中判断真正的独立方向。

第三，CS229 的基础 ICA 模型假设 $A\in\mathbb R^{d\times d}$ 且可逆。这意味着源信号数量等于观测信号数量，在鸡尾酒会问题中就是说话人数等于麦克风数量。

## 4. ICA 的固有不确定性

即使 ICA 成功恢复出了源信号，也仍然有一些东西无法确定。这些不是算法缺陷，而是模型本身导致的不可辨识性。

首先是顺序不确定性。ICA 可能恢复出 $s_1,s_2,\dots,s_d$，也可能恢复出重新排序后的版本。数学上，如果 $P$ 是置换矩阵，那么 $Ps$ 只是把源信号顺序重新排列。由于 $x=As$ 也可以写成 $x=AP^{-1}Ps$，仅凭观测数据无法判断原始源信号应该按什么顺序排列。

其次是尺度和符号不确定性。如果把某个源信号 $s_j$ 放大 $\alpha_j$ 倍，同时把混合矩阵 $A$ 的第 $j$ 列缩小 $\alpha_j$ 倍，那么乘积 $As$ 不变。因此 ICA 无法确定源信号的真实振幅大小。符号也无法确定，因为 $s_j$ 和 $-s_j$ 只差一个正负号；对于声音信号而言，波形整体乘以 $-1$ 后，听起来通常没有本质差别。

所以 ICA 最终恢复出的源信号通常只在顺序、尺度和符号意义下确定。

## 5. 密度变换

为了用最大似然估计来求解 $W$，需要写出观测数据 $x$ 的概率密度 $p_x(x)$。但 ICA 的假设是关于源信号 $s$ 的密度 $p_s(s)$，所以需要建立二者关系。

模型中 $s=Wx$，所以 $s$ 是 $x$ 的线性变换。一维情况下，如果 $s=wx$，概率守恒给出：

$$
p_x(x)|dx|=p_s(s)|ds|
$$

两边除以 $|dx|$，得到：

$$
p_x(x)=p_s(s)\left|\frac{ds}{dx}\right|
$$

由于 $s=wx$，所以 $p_x(x)=p_s(wx)|w|$。这里的 $|w|$ 不能省略，它表示坐标变换造成的长度缩放。

高维情况下，长度缩放变成体积缩放。线性变换 $s=Wx$ 会使体积微元按雅可比行列式的绝对值缩放，即 $dV_s=|\det W|dV_x$。由概率守恒：

$$
p_x(x)dV_x=p_s(s)dV_s
$$

代入 $dV_s=|\det W|dV_x$：

$$
p_x(x)dV_x=p_s(s)|\det W|dV_x
$$

所以：

$$
p_x(x)=p_s(Wx)|\det W|
$$

这就是 ICA 中最关键的密度变换公式。不能只写 $p_x(x)=p_s(Wx)$，还必须乘上 $|\det W|$，因为线性变换会改变体积，从而改变密度。

## 6. 利用独立性写出似然

由于源信号各个分量独立，有 $p_s(s)=\prod_{j=1}^d p_j(s_j)$。而 $s_j=w_j^Tx$，所以：

$$
p_s(Wx)=\prod_{j=1}^d p_j(w_j^Tx)
$$

代入密度变换公式：

$$
p_x(x)=\left(\prod_{j=1}^d p_j(w_j^Tx)\right)|\det W|
$$

这就是 ICA 对观测数据 $x$ 的概率建模。它的含义是：先用 $W$ 把观测信号 $x$ 映射回源空间，再假设映射后的每个分量 $w_j^Tx$ 彼此独立，最后乘上体积变换项 $|\det W|$。

对于训练集 $\{x^{(1)},x^{(2)},\dots,x^{(m)}\}$，log-likelihood 为：

$$
\ell(W)=
\sum_{i=1}^m
\left[
\sum_{j=1}^d \log p_j(w_j^Tx^{(i)})
+
\log|\det W|
\right]
$$

ICA 的最大似然估计就是寻找一个 $W$，让恢复出的源信号既符合假设的非高斯源分布，又让整体观测数据的似然最大。

## 7. 为什么高斯源不适合 ICA

PS4 的第一问要求解释为什么 Gaussian source 会出问题。假设源信号服从标准正态分布，即 $s\sim\mathcal N(0,I)$。由于 $x=As$，所以 $x$ 也是高斯分布：

$$
x\sim\mathcal N(0,AA^T)
$$

这说明从观测数据中，我们最多能识别出协方差矩阵 $AA^T$。但这不足以唯一确定 $A$。

原因是：对任意正交矩阵 $R$，都有 $RR^T=I$。如果把混合矩阵换成 $A'=AR$，则：

$$
A'{A'}^T=(AR)(AR)^T=ARR^TA^T=AA^T
$$

也就是说，$A$ 和 $AR$ 会产生完全相同的观测协方差。对于高斯分布来说，协方差已经决定了整个分布，因此这两种混合方式在概率分布上无法区分。

从 $W$ 的角度也一样。高斯源下，log-likelihood 只会要求变换后的源具有单位协方差，大致约束为：

$$
W\left(\frac1mX^TX\right)W^T=I
$$

这只是在做 whitening。满足这个条件的 $W$ 不是唯一的，因为 whitening 后还可以再乘任意正交旋转矩阵。这个旋转自由度无法通过高斯分布消除。

所以，高斯源不适合 ICA 的根本原因是：高斯分布具有旋转不变性，ICA 无法从高斯源中识别出唯一的独立方向。

下面把这个结论从最大似然的角度推一遍。高斯源假设下，单个源分量的密度是 $p(s_j)=\frac{1}{\sqrt{2\pi}}\exp(-\frac12s_j^2)$。又因为 $s_j=w_j^Tx^{(i)}$，所以训练集的 log-likelihood 为：

$$
\ell(W)
=
\sum_{i=1}^m
\left[
\log|W|
+
\sum_{j=1}^d
\log\left(
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac12(w_j^Tx^{(i)})^2\right)
\right)
\right]
$$

把和 $W$ 无关的常数 $\log\frac{1}{\sqrt{2\pi}}$ 去掉：

$$
\ell(W)
=
m\log|W|
-
\frac12\sum_{i=1}^m\sum_{j=1}^d(w_j^Tx^{(i)})^2
+\text{constant}
$$

其中：

$$
\sum_{j=1}^d(w_j^Tx^{(i)})^2
=
\|Wx^{(i)}\|_2^2
=
{x^{(i)}}^TW^TWx^{(i)}
$$

对 $W$ 求梯度。第一项直接用矩阵基础中的公式：

$$
\nabla_W[m\log|W|]=m(W^{-1})^T
$$

第二项可以写成：

$$
-\frac12\sum_{i=1}^m\|Wx^{(i)}\|_2^2
$$

对单个样本有：

$$
\nabla_W\left[\frac12\|Wx^{(i)}\|_2^2\right]
=
Wx^{(i)}{x^{(i)}}^T
$$

因此：

$$
\nabla_W\ell(W)
=
m(W^{-1})^T
-
\sum_{i=1}^mWx^{(i)}{x^{(i)}}^T
$$

把 $W$ 提出来：

$$
\nabla_W\ell(W)
=
m(W^{-1})^T
-
W\sum_{i=1}^m x^{(i)}{x^{(i)}}^T
$$

如果数据矩阵 $X\in\mathbb R^{m\times d}$ 按行存样本，则 $\sum_i x^{(i)}{x^{(i)}}^T=X^TX$。所以：

$$
\nabla_W\ell(W)
=
m(W^{-1})^T-WX^TX
$$

令梯度为 0：

$$
m(W^{-1})^T-WX^TX=0
$$

于是：

$$
m(W^{-1})^T=WX^TX
$$

左乘 $W^T$：

$$
mI=W^TWX^TX
$$

右乘 $(X^TX)^{-1}$：

$$
W^TW=m(X^TX)^{-1}
$$

这个式子说明，高斯源下最大似然只能确定 $W^TW$，也就是只能确定一种 whitening 约束，而不能唯一确定 $W$ 本身。因为如果某个 $W$ 满足条件，那么对任意正交矩阵 $R$，$RW$ 也会产生同样的独立标准高斯源分布。这就是 Gaussian source 的旋转不确定性。

## 8. Laplace 源分布下的 ICA

PS4 的第二问要求假设源信号服从标准 Laplace 分布。标准 Laplace 分布的密度为：

$$
p(s_j)=\frac12\exp(-|s_j|)
$$

对单个样本 $x$，由 ICA 的似然公式：

$$
\ell(W)=\log|\det W|+\sum_{j=1}^d\log p_j(w_j^Tx)
$$

代入 Laplace 密度：

$$
\ell(W)=\log|\det W|+\sum_{j=1}^d\log\left(\frac12\exp(-|w_j^Tx|)\right)
$$

展开：

$$
\ell(W)=\log|\det W|+\sum_{j=1}^d\left[-\log2-|w_j^Tx|\right]
$$

去掉与 $W$ 无关的常数 $-\log2$：

$$
\ell(W)=\log|\det W|-\sum_{j=1}^d|w_j^Tx|
$$

现在对 $W$ 求梯度。第一项直接用矩阵基础中的公式：

$$
\nabla_W\log|\det W|=(W^T)^{-1}
$$

第二项看单个 $j$。令 $z_j=w_j^Tx$，则 $\nabla_{w_j}|w_j^Tx|=\operatorname{sign}(w_j^Tx)x$。所以：

$$
\nabla_{w_j}\left[-|w_j^Tx|\right]
=
-\operatorname{sign}(w_j^Tx)x
$$

把所有行堆成矩阵形式：

$$
\nabla_W\left[-\sum_{j=1}^d|w_j^Tx|\right]
=
-\operatorname{sign}(Wx)x^T
$$

因此单样本 log-likelihood 的梯度是：

$$
\nabla_W\ell(W)
=
(W^T)^{-1}
-
\operatorname{sign}(Wx)x^T
$$

使用随机梯度上升，更新规则为：

$$
W:=W+\alpha\left((W^T)^{-1}-\operatorname{sign}(Wx)x^T\right)
$$

这就是 PS4 part (b) 中 Laplace source 的 ICA 更新公式。

直觉上，$\log|\det W|$ 来自密度变换中的体积项，防止 $W$ 退化；而 $-\sum_j|w_j^Tx|$ 来自 Laplace 源分布，它鼓励恢复出的源信号符合 Laplace 的非高斯形状。

## 9. Logistic 源分布与讲义公式

CS229 讲义中使用的是 logistic 分布。讲义把 sigmoid 函数 $g(s)=\frac{1}{1+e^{-s}}$ 作为源信号的 CDF，因此源信号密度是 $g'(s)$。这时单样本 log-likelihood 为：

$$
\ell(W)=\log|\det W|+\sum_{j=1}^d\log g'(w_j^Tx)
$$

对应的随机梯度上升更新会出现 $\frac{d}{dz}\log g'(z)=1-2g(z)$，因此有讲义里的更新形式：

$$
W:=W+\alpha\left(
\begin{bmatrix}
1-2g(w_1^Tx)\\
1-2g(w_2^Tx)\\
\vdots\\
1-2g(w_d^Tx)
\end{bmatrix}
x^T
+
(W^T)^{-1}
\right)
$$

这和 Laplace 版本的结构是一样的：一部分来自源分布的 log-density，一部分来自 $\log|\det W|$。PS4 代码题只是把讲义中的 logistic source 换成了 Laplace source。

## 10. 训练完成后的源信号恢复

当 $W$ 通过梯度上升训练收敛后，就可以对每个观测样本计算 $s^{(i)}=Wx^{(i)}$，得到恢复出的独立源信号。

在代码里，训练集通常按行存样本，$X\in\mathbb R^{m\times d}$。如果数学上单样本是列向量 $s=Wx$，那么按行堆起来，则有$S=[--s^{(i)T}--]$堆叠，$s^{(i)T}=x^{(i)T}W^T$, 展开后为 $S=XW^T$。也可以看看维度：输入 $X$ 是 $(m,d)$，输出的源信号矩阵 $S$ 也应该是 $(m,d)$。

一下是拉普拉斯分布时，W的更新
```python
updated_W=W+learning_rate*(np.linalg.inv(W).T-np.sign(W@x)@x.T)
```
需要注意，恢复出的源信号可能存在顺序变化、音量尺度变化和整体正负号变化，但这些通常不影响盲源分离的主要目标。

## 11. ICA 和 PCA 的对比

PCA 与 ICA 都可以看成是在寻找新的表示方式，但目标完全不同。PCA 的目标是找到最大方差的正交方向，它只关心二阶统计量，也就是协方差矩阵。PCA 得到的新分量之间不相关，但不一定独立。

ICA 的目标是找到统计独立的源成分。它不仅要去除线性相关，还要尽可能去除更深层的统计依赖。因此，PCA 是去相关，不保证独立；ICA 是追求独立，并且依赖非高斯性。

另外，PCA 的主成分通常可以按方差大小排序；ICA 的独立成分一般没有天然顺序。

