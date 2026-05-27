# EM 算法与高斯混合模型

关联：[[数学基础知识]]、[[数学知识/概率论基础]]、[[数学知识/矩阵基础]]

这页整理 EM 算法和 GMM。主线是：有隐变量时，直接做 MLE 会得到互相依赖的固定点方程；EM 通过 Jensen 不等式构造下界，然后交替进行 E-step 和 M-step。

## 1. GMM 的基本思想

GMM 的全称是 Gaussian Mixture Model，即高斯混合模型。它的核心想法是：真实数据不一定来自一个单独的高斯分布，而可能是由多个高斯分布混合生成的。

![[GMM.png|Pasted image 20260527145609.png]]

例如一堆身高数据可能同时包含男生和女生。男生身高大致服从一个高斯分布 (红线，均值更高)，女生身高大致服从另一个高斯分布 （黑线，均值比男生低）。我们观测到的是混在一起的身高 $x$ (黑线)，但看不到每个样本到底来自哪个群体。

在 GMM 中，每个样本背后都有一个隐藏的类别变量 $z^{(i)}$，$z^{(i)}=j$ 表示第 $i$ 个样本来自第 $j$ 个高斯分布。如果一共有 $k$ 个高斯组分，则 $z^{(i)}\in\{1,2,\dots,k\}$。模型可以写成 $z^{(i)}\sim \mathrm{Multinomial}(\phi)$。如果 $z^{(i)}=j$，则 $x^{(i)}\mid z^{(i)}=j\sim \mathcal N(\mu_j,\Sigma_j)$。

所以一个样本的边缘分布是多个高斯分布的加权和：

$$
p(x^{(i)};\theta)=\sum_{j=1}^k\phi_j\mathcal N(x^{(i)};\mu_j,\Sigma_j)
$$

这里的 $\phi_j$ 表示第 $j$ 个高斯组分出现的概率，满足 $\sum_{j=1}^k\phi_j=1$。GMM 的难点在于：如果知道每个样本的 $z^{(i)}$，那参数估计会很简单；但现实中我们通常只看到 $x^{(i)}$，看不到它属于哪个高斯组分。因此，GMM 自然引出了 EM 算法：E-step 先根据当前参数估计每个样本属于各个高斯组分的概率，M-step 再把这些概率当作权重，重新估计每个高斯组分的参数。

## 2. 为什么需要 EM

在普通 MLE 中，如果观测数据完整，我们通常可以直接写出 likelihood，然后对参数求导。但有些问题中，数据背后存在没有观测到的变量，称为隐变量 latent variable。此时直接最大化 likelihood 往往会出现“参数依赖隐变量，隐变量又依赖参数”的循环。

### 2.1 男女身高混合例子

假设我们收集了 $m$ 个学生的身高数据 $\{x^{(1)},x^{(2)},\dots,x^{(m)}\}$，但不知道每个学生的性别。令隐变量 $z^{(i)}$ 表示第 $i$ 个样本来自哪个群体：$z^{(i)}=1$ 表示男生，$z^{(i)}=2$ 表示女生。

假设：

$x\mid z=1\sim \mathcal N(\mu_1,\sigma^2)$。

$x\mid z=2\sim \mathcal N(\mu_2,\sigma^2)$。

$p(z=1)=\phi$，$p(z=2)=1-\phi$。

那么单个身高样本的边缘概率是：

$$
p(x^{(i)};\theta)=\phi\mathcal N(x^{(i)};\mu_1,\sigma^2)+(1-\phi)\mathcal N(x^{(i)};\mu_2,\sigma^2)
$$

总对数似然为：

$$
\ell(\theta)=\sum_{i=1}^m\log\left[\phi\mathcal N(x^{(i)};\mu_1,\sigma^2)+(1-\phi)\mathcal N(x^{(i)};\mu_2,\sigma^2)\right]
$$

这里最大的麻烦是 $\log$ 里面有一个求和。相比完整标签情况，这个式子很难直接求出闭式解。

### 2.2 直接求导为什么会卡住

以 $\mu_1$ 为例，对 $\ell(\theta)$ 求导。

记 $\mathcal N_1^{(i)}=\mathcal N(x^{(i)};\mu_1,\sigma^2)$，$\mathcal N_2^{(i)}=\mathcal N(x^{(i)};\mu_2,\sigma^2)$。

则：

$\frac{\partial \ell}{\partial \mu_1}=\sum_{i=1}^m\frac{1}{\phi\mathcal N_1^{(i)}+(1-\phi)\mathcal N_2^{(i)}}\phi\frac{\partial \mathcal N_1^{(i)}}{\partial \mu_1}$。

正态分布对均值求导有：

$\frac{\partial \mathcal N(x;\mu,\sigma^2)}{\partial \mu}=\mathcal N(x;\mu,\sigma^2)\frac{x-\mu}{\sigma^2}$。

代入可得：

$\frac{\partial \ell}{\partial \mu_1}=\sum_{i=1}^m\frac{\phi\mathcal N_1^{(i)}}{\phi\mathcal N_1^{(i)}+(1-\phi)\mathcal N_2^{(i)}}\frac{x^{(i)}-\mu_1}{\sigma^2}$。

令：

$w_1^{(i)}=\frac{\phi\mathcal N(x^{(i)};\mu_1,\sigma^2)}{\phi\mathcal N(x^{(i)};\mu_1,\sigma^2)+(1-\phi)\mathcal N(x^{(i)};\mu_2,\sigma^2)}$。

于是：

$\frac{\partial \ell}{\partial \mu_1}=\sum_{i=1}^m w_1^{(i)}\frac{x^{(i)}-\mu_1}{\sigma^2}$。

令梯度为 0：

$\sum_{i=1}^m w_1^{(i)}(x^{(i)}-\mu_1)=0$。

展开：

$\sum_{i=1}^m w_1^{(i)}x^{(i)}-\mu_1\sum_{i=1}^m w_1^{(i)}=0$。

所以：

$$
\mu_1=\frac{\sum_{i=1}^m w_1^{(i)}x^{(i)}}{\sum_{i=1}^m w_1^{(i)}}
$$

这个式子看起来像闭式解，但其实不是，因为 $w_1^{(i)}$ 本身又依赖 $\mu_1,\mu_2,\phi$。所以直接 MLE 并不是完全不能求导，而是求导后得到的是互相依赖的固定点方程：要更新 $\mu_1$，需要先知道每个样本属于第 1 类的概率 $w_1^{(i)}$；但要计算 $w_1^{(i)}$，又需要先知道 $\mu_1,\mu_2,\phi$。这就是 EM 要解决的鸡生蛋问题。

## 3. EM 的核心想法

EM 不直接最大化难处理的 $\ell(\theta)$，而是先为它构造一个容易优化的下界，然后重复两步：E-step 固定当前参数 $\theta^{(t)}$，估计隐变量的后验分布；M-step 固定 E-step 算出的后验分布，最大化下界，得到新参数 $\theta^{(t+1)}$。直觉上，E-step 是“根据当前模型猜标签”，M-step 是“把猜出来的软标签当权重，重新做 MLE”。

## 4. Jensen 不等式与下界

EM 的数学基础是 Jensen 不等式。

![[Jensen.png|Pasted image 20260527150030.png]]

若 $f$ 是凹函数，则 $\mathbb E[f(X)]\le f(\mathbb E[X])$。因为 $\log x$ 是凹函数，所以 $\mathbb E[\log X]\le \log\mathbb E[X]$，也就是 $\log\mathbb E[X]\ge \mathbb E[\log X]$。这正好可以把难处理的 $\log\sum$ 变成一个容易处理的“期望里的 log”。

## 5. EM 下界推导

对于单个样本，边缘 likelihood 是 $\log p(x^{(i)};\theta)=\log\sum_{z^{(i)}=1}^k p(x^{(i)},z^{(i)};\theta)$。引入任意分布 $Q_i(z^{(i)})$，满足 $\sum_{z^{(i)}=1}^k Q_i(z^{(i)})=1$，则：

$$
\log p(x^{(i)};\theta)=\log\sum_{z^{(i)}=1}^k Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

这可以看成对 $z^{(i)}\sim Q_i$ 的期望：

$$
\log p(x^{(i)};\theta)=\log\mathbb E_{z^{(i)}\sim Q_i}\left[\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]
$$

由 Jensen 不等式：

$$
\log\mathbb E_{z^{(i)}\sim Q_i}\left[\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]
\ge
\mathbb E_{z^{(i)}\sim Q_i}\left[\log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]
$$

展开期望：

$$
\log p(x^{(i)};\theta)
\ge
\sum_{z^{(i)}=1}^k Q_i(z^{(i)})\log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

对所有样本求和：

$$
\ell(\theta)
\ge
\sum_{i=1}^m\sum_{z^{(i)}=1}^k Q_i(z^{(i)})\log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

记右侧为下界函数：

$$
J(Q,\theta)=\sum_{i=1}^m\sum_{z^{(i)}=1}^k Q_i(z^{(i)})\log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

因此：

$$
\ell(\theta)\ge J(Q,\theta)
$$

## 6. EM 过程

EM 可以简化成这个图片所示的过程：E-step 让下界贴住原函数，M-step 固定 $Q$ 后更新 $\theta$。

![[EM算法.png|Pasted image 20260527150516.png]]

### 6.1 E-step：让下界贴住原函数

E-step 固定当前参数 $\theta^{(t)}$，选择 $Q_i$，使下界 $J(Q,\theta^{(t)})$ 尽可能贴近真实 likelihood $\ell(\theta^{(t)})$。

Jensen 等号成立要求：

$$
\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{Q_i(z^{(i)})}
$$

对不同的 $z^{(i)}$ 都是同一个常数。所以 $Q_i(z^{(i)})\propto p(x^{(i)},z^{(i)};\theta^{(t)})$。

归一化：

$$
Q_i(z^{(i)})
=
\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{\sum_z p(x^{(i)},z;\theta^{(t)})}
$$

由条件概率公式：

$$
Q_i(z^{(i)})=p(z^{(i)}\mid x^{(i)};\theta^{(t)})
$$

这就是 E-step。在 GMM 里，$Q_i(z^{(i)}=j)$ 常写成 responsibility：

$$
w_j^{(i)}=p(z^{(i)}=j\mid x^{(i)};\theta^{(t)})
$$

### 6.2 M-step：固定软标签，重新估计参数

M-step 固定 E-step 算出的 $Q_i$，去最大化下界：

$$
\theta^{(t+1)}
=
\arg\max_\theta
\sum_{i=1}^m\sum_{z^{(i)}=1}^k
Q_i(z^{(i)})\log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

因为 $Q_i(z^{(i)})$ 在 M-step 中是常数，且 $-\log Q_i(z^{(i)})$ 与 $\theta$ 无关，所以 M-step 等价于：

$$
\theta^{(t+1)}
=
\arg\max_\theta
\sum_{i=1}^m\sum_{z^{(i)}=1}^k
Q_i(z^{(i)})\log p(x^{(i)},z^{(i)};\theta)
$$

注意：M-step 求导时，$Q_i(z^{(i)})$ 或 $w_j^{(i)}$ 当作常数。它们依赖旧参数 $\theta^{(t)}$，但不依赖这一步正在求的新参数 $\theta$。这点很重要，否则就会误以为 $w_j^{(i)}$ 也要一起求导。

## 7. 半监督 EM 的目标函数

半监督 EM 的场景是：一部分样本没有标签，一部分样本有标签。无标签样本写作 $\{x^{(1)},x^{(2)},\dots,x^{(m)}\}$，有标签样本写作：

$$
\{(\tilde x^{(1)},\tilde z^{(1)}),(\tilde x^{(2)},\tilde z^{(2)}),\dots,(\tilde x^{(\tilde m)},\tilde z^{(\tilde m)})\}
$$

其中 $z^{(i)}$ 是无标签样本背后的隐变量，$\tilde z^{(i)}$ 是有标签样本的已知类别。半监督目标函数是：

$$
\ell_{\mathrm{semi}}(\theta)=\ell_{\mathrm{unsup}}(\theta)+\alpha\ell_{\mathrm{sup}}(\theta)
$$

其中：

$$
\ell_{\mathrm{unsup}}(\theta)=\sum_{i=1}^m\log\sum_{z^{(i)}=1}^k p(x^{(i)},z^{(i)};\theta)
$$

$$
\ell_{\mathrm{sup}}(\theta)=\sum_{i=1}^{\tilde m}\log p(\tilde x^{(i)},\tilde z^{(i)};\theta)
$$

$\alpha$ 是有标签样本的权重。如果 $\alpha$ 大，模型会更相信有标签数据。半监督 EM 的 E-step 只对无标签样本估计隐变量 $w_j^{(i)}=p(z^{(i)}=j\mid x^{(i)};\theta^{(t)})$。有标签样本的 $\tilde z^{(i)}$ 已经知道，所以不需要估计。半监督 M-step 最大化：

$$
\sum_{i=1}^m\sum_{j=1}^k w_j^{(i)}\log p(x^{(i)},z^{(i)}=j;\theta)
+
\alpha\sum_{i=1}^{\tilde m}\log p(\tilde x^{(i)},\tilde z^{(i)};\theta)
$$

对于 GMM，有：

$$
p(x^{(i)},z^{(i)}=j;\theta)=\phi_j\mathcal N(x^{(i)};\mu_j,\Sigma_j)
$$

有标签部分也可以写成指示函数形式：

$$
\log p(\tilde x^{(i)},\tilde z^{(i)};\theta)
=
\sum_{j=1}^k1\{\tilde z^{(i)}=j\}\log\left[\phi_j\mathcal N(\tilde x^{(i)};\mu_j,\Sigma_j)\right]
$$

因此，半监督 GMM 的 M-step 目标可以写成：

$$
\mathcal L(\theta)
=
\sum_{i=1}^m\sum_{j=1}^k w_j^{(i)}\log\left[\phi_j\mathcal N(x^{(i)};\mu_j,\Sigma_j)\right]
+
\alpha\sum_{i=1}^{\tilde m}\sum_{j=1}^k1\{\tilde z^{(i)}=j\}\log\left[\phi_j\mathcal N(\tilde x^{(i)};\mu_j,\Sigma_j)\right]
$$

接下来分别对 $\phi_j,\mu_j,\Sigma_j$ 求最大值。

## 8. 半监督 GMM 的 E-step

E-step 要估计的 latent variables 只有无标签样本的 $z^{(i)}$。由贝叶斯公式：

$$
w_j^{(i)}
=
p(z^{(i)}=j\mid x^{(i)};\phi,\mu,\Sigma)
$$

展开：

$$
w_j^{(i)}
=
\frac{p(x^{(i)}\mid z^{(i)}=j;\mu_j,\Sigma_j)p(z^{(i)}=j;\phi)}
{\sum_{l=1}^k p(x^{(i)}\mid z^{(i)}=l;\mu_l,\Sigma_l)p(z^{(i)}=l;\phi)}
$$

代入 GMM：

$$
w_j^{(i)}
=
\frac{\phi_j\mathcal N(x^{(i)};\mu_j,\Sigma_j)}
{\sum_{l=1}^k\phi_l\mathcal N(x^{(i)};\mu_l,\Sigma_l)}
$$

再把多元高斯密度写开：

$$
w_j^{(i)}
=
\frac{
\phi_j\frac{1}{(2\pi)^{n/2}|\Sigma_j|^{1/2}}
\exp\left[-\frac12(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)\right]
}{
\sum_{l=1}^k
\phi_l\frac{1}{(2\pi)^{n/2}|\Sigma_l|^{1/2}}
\exp\left[-\frac12(x^{(i)}-\mu_l)^T\Sigma_l^{-1}(x^{(i)}-\mu_l)\right]
}
$$

注意：$w_j^{(i)}$ 在 E-step 中由旧参数算出；进入 M-step 后，它被当作常数。

## 9. 半监督 GMM 的 M-step

### 9.1 更新 $\phi_j$

先看 $\phi_j$。
$$
\mathcal L(\theta)
=
\sum_{i=1}^m\sum_{j=1}^k w_j^{(i)}\log\left[\phi_j\mathcal N(x^{(i)};\mu_j,\Sigma_j)\right]
+
\alpha\sum_{i=1}^{\tilde m}\sum_{j=1}^k1\{\tilde z^{(i)}=j\}\log\left[\phi_j\mathcal N(\tilde x^{(i)};\mu_j,\Sigma_j)\right]
$$

在 $\mathcal L(\theta)$ 中，和 $\phi_j$ 有关的只有 $\log[\phi_j p(x^{(i)}\mid z^{(i)}=j)]$ 里的 $\log\phi_j$。因为高斯密度里的 $\mu_j,\Sigma_j$ 与 $\phi_j$ 无关，所以取出所有和 $\phi$ 有关的项：

$$
\mathcal L(\phi)
=
\sum_{i=1}^m\sum_{j=1}^k w_j^{(i)}\log\phi_j
+
\alpha\sum_{i=1}^{\tilde m}\sum_{j=1}^k1\{\tilde z^{(i)}=j\}\log\phi_j
$$

因为 $\phi$ 是类别概率，还要满足约束 $\sum_{j=1}^k\phi_j=1$。加入拉格朗日乘子 $\beta$：

$$
\mathcal L(\phi)
=
\sum_{i=1}^m\sum_{j=1}^k w_j^{(i)}\log\phi_j
+
\alpha\sum_{i=1}^{\tilde m}\sum_{j=1}^k1\{\tilde z^{(i)}=j\}\log\phi_j
+
\beta\left(\sum_{j=1}^k\phi_j-1\right)
$$

对某个 $\phi_j$ 求导：

$$
\frac{\partial\mathcal L}{\partial\phi_j}
=
\frac{\sum_{i=1}^m w_j^{(i)}}{\phi_j}
+
\frac{\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}}{\phi_j}
+
\beta
$$

令其为 0：

$$
\frac{
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}{\phi_j}
+
\beta
=0
$$

所以：

$$
\phi_j
=
-
\frac{
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}{\beta}
$$

对 $j$ 求和：

$$
\sum_{j=1}^k\phi_j
=
-
\frac{
\sum_{j=1}^k\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{j=1}^k\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}{\beta}
$$

因为每个无标签样本对所有类别的 responsibility 加起来为 1，即 $\sum_{j=1}^k w_j^{(i)}=1$，所以：

$$
\sum_{j=1}^k\sum_{i=1}^m w_j^{(i)}=m
$$

每个有标签样本只属于一个类别，所以 $\sum_{j=1}^k1\{\tilde z^{(i)}=j\}=1$。因此：

$$
\sum_{j=1}^k\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}=\tilde m
$$

代回约束 $\sum_j\phi_j=1$：

$$
1=-\frac{m+\alpha\tilde m}{\beta}
$$

所以：

$$
\beta=-(m+\alpha\tilde m)
$$

最终：

$$
\phi_j
=
\frac{
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}{
m+\alpha\tilde m
}
$$

直觉上，$\phi_j$ 是第 $j$ 类的样本比例。无标签样本用软计数 $w_j^{(i)}$，有标签样本用硬计数 $1\{\tilde z^{(i)}=j\}$。

### 9.2 更新 $\mu_j$

现在看 $\mu_j$。在 $\mathcal L(\theta)$ 中，和 $\mu_j$ 有关的只有第 $j$ 个高斯分布里的二次型：

$$
\log\mathcal N(x;\mu_j,\Sigma_j)
=
-\frac12\log|\Sigma_j|
-\frac12(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
-\frac n2\log(2\pi)
$$

其中 $\log|\Sigma_j|$ 和 $\frac n2\log(2\pi)$ 与 $\mu_j$ 无关，所以只需要看：

$$
-\frac12(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
$$

对 $\mu_j$ 求导有：

$$
\nabla_{\mu_j}\left[-\frac12(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)\right]
=
\Sigma_j^{-1}(x-\mu_j)
$$

因此无标签部分：

$$
\nabla_{\mu_j}\ell_{\mathrm{unsup}}
=
\sum_{i=1}^m w_j^{(i)}\Sigma_j^{-1}(x^{(i)}-\mu_j)
$$

提出 $\Sigma_j^{-1}$：

$$
\nabla_{\mu_j}\ell_{\mathrm{unsup}}
=
\Sigma_j^{-1}\left(
\sum_{i=1}^m w_j^{(i)}x^{(i)}
-
\mu_j\sum_{i=1}^m w_j^{(i)}
\right)
$$

有标签部分中，只有满足 $\tilde z^{(i)}=j$ 的样本和 $\mu_j$ 有关：

$$
\nabla_{\mu_j}\ell_{\mathrm{sup}}
=
\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\Sigma_j^{-1}(\tilde x^{(i)}-\mu_j)
$$

提出 $\Sigma_j^{-1}$：

$$
\nabla_{\mu_j}\ell_{\mathrm{sup}}
=
\Sigma_j^{-1}\left(
\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\tilde x^{(i)}
-
\mu_j\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
$$

半监督目标的梯度是两者相加：

$$
\nabla_{\mu_j}\ell_{\mathrm{semi}}
=
\nabla_{\mu_j}\ell_{\mathrm{unsup}}
+
\alpha\nabla_{\mu_j}\ell_{\mathrm{sup}}
$$

代入：

$$
\nabla_{\mu_j}\ell_{\mathrm{semi}}
=
\Sigma_j^{-1}
\left[
\left(\sum_{i=1}^m w_j^{(i)}x^{(i)}-\mu_j\sum_{i=1}^m w_j^{(i)}\right)
+
\alpha
\left(\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\tilde x^{(i)}-\mu_j\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\right)
\right]
$$

整理：

$$
\nabla_{\mu_j}\ell_{\mathrm{semi}}
=
\Sigma_j^{-1}
\left[
\sum_{i=1}^m w_j^{(i)}x^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\tilde x^{(i)}
-
\mu_j
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\right]
$$

令梯度为 0。由于 $\Sigma_j^{-1}$ 可逆，括号内为 0：

$$
\sum_{i=1}^m w_j^{(i)}x^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\tilde x^{(i)}
-
\mu_j
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
=0
$$

解出：

$$
\mu_j
=
\frac{
\sum_{i=1}^m w_j^{(i)}x^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\tilde x^{(i)}
}{
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}
$$

直觉上，$\mu_j$ 是第 $j$ 类样本的加权平均。无标签数据用软权重，有标签数据用硬权重。

### 9.3 更新 $\Sigma_j$

最后看 $\Sigma_j$。在 $\mathcal L(\theta)$ 中，和 $\Sigma_j$ 有关的是第 $j$ 个高斯分布里的两项：

$$
-\frac12\log|\Sigma_j|
$$

以及：

$$
-\frac12(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)
$$

先取出无标签部分中和 $\Sigma_j$ 有关的项：

$$
\ell_{\mathrm{unsup}}(\Sigma_j)
=
-\frac12\sum_{i=1}^m w_j^{(i)}\log|\Sigma_j|
-\frac12\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)
$$

有标签部分中，只有 $\tilde z^{(i)}=j$ 的样本和 $\Sigma_j$ 有关：

$$
\ell_{\mathrm{sup}}(\Sigma_j)
=
-\frac12\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\log|\Sigma_j|
-\frac12\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)^T\Sigma_j^{-1}(\tilde x^{(i)}-\mu_j)
$$

半监督目标里，$\ell_{\mathrm{sup}}$ 前面有权重 $\alpha$，所以：

$$
\ell_{\mathrm{semi}}(\Sigma_j)
=
-\frac12\sum_{i=1}^m w_j^{(i)}\log|\Sigma_j|
-\frac12\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)
$$

$$
\quad
-\frac12\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\log|\Sigma_j|
-\frac12\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)^T\Sigma_j^{-1}(\tilde x^{(i)}-\mu_j)
$$

现在对 $\Sigma_j$ 求导。第一类项是 $\log|\Sigma_j|$，这里直接使用 [[数学知识/矩阵基础]] 中的公式：

$$
\nabla_{\Sigma_j}\log|\Sigma_j|=\Sigma_j^{-1}
$$

所以 $\log|\Sigma_j|$ 这部分的梯度是：

$$
\nabla_{\Sigma_j}
\left[
-\frac12\sum_{i=1}^m w_j^{(i)}\log|\Sigma_j|
-\frac12\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}\log|\Sigma_j|
\right]
$$

$$
=
-\frac12
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\Sigma_j^{-1}
$$

第二类项是二次型。先看一个通用形式，令 $c=x-\mu_j$，考虑：

$$
c^T\Sigma_j^{-1}c
$$

它是标量，可以写成 trace：

$$
c^T\Sigma_j^{-1}c
=
\mathrm{tr}(c^T\Sigma_j^{-1}c)
=
\mathrm{tr}(\Sigma_j^{-1}cc^T)
$$

对它求微分：

$$
d(c^T\Sigma_j^{-1}c)
=
d\,\mathrm{tr}(\Sigma_j^{-1}cc^T)
$$

因为这里对 $\Sigma_j$ 求导，所以 $c=x-\mu_j$ 暂时看作常数：

$$
d\,\mathrm{tr}(\Sigma_j^{-1}cc^T)
=
\mathrm{tr}(d(\Sigma_j^{-1})cc^T)
$$

用[[矩阵基础#4. Trace 技巧]]        $d(\Sigma_j^{-1})=-\Sigma_j^{-1}(d\Sigma_j)\Sigma_j^{-1}$：

$$
d(c^T\Sigma_j^{-1}c)
=
-\mathrm{tr}(\Sigma_j^{-1}(d\Sigma_j)\Sigma_j^{-1}cc^T)
$$

用 trace 循环置换，把 $d\Sigma_j$ 放到最后：

$$
d(c^T\Sigma_j^{-1}c)
=
-\mathrm{tr}(\Sigma_j^{-1}cc^T\Sigma_j^{-1}d\Sigma_j)
$$

因此：

$$
\nabla_{\Sigma_j}\left(c^T\Sigma_j^{-1}c\right)
=
-\Sigma_j^{-1}cc^T\Sigma_j^{-1}
$$

所以单个无标签样本的二次型项：

$$
\nabla_{\Sigma_j}\left[-\frac12w_j^{(i)}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)\right]
=
\frac12w_j^{(i)}\Sigma_j^{-1}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T\Sigma_j^{-1}
$$

有标签样本同理，只是权重变成 $\alpha1\{\tilde z^{(i)}=j\}$。把 $\log|\Sigma_j|$ 项和二次型项合起来：

$$
\nabla_{\Sigma_j}\ell_{\mathrm{semi}}
=
-\frac12
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\Sigma_j^{-1}
$$

$$
\quad
+
\frac12
\Sigma_j^{-1}
\left(
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
\right)
\Sigma_j^{-1}
$$

令梯度为 0：

$$
-\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\Sigma_j^{-1}
$$

$$
+
\Sigma_j^{-1}
\left(
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
\right)
\Sigma_j^{-1}
=0
$$

移项：

$$
\Sigma_j^{-1}
\left(
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
\right)
\Sigma_j^{-1}
$$

$$
=
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\Sigma_j^{-1}
$$

左乘 $\Sigma_j$：

$$
\left(
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
\right)
\Sigma_j^{-1}
$$

$$
=
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
I
$$

右乘 $\Sigma_j$：

$$
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
$$

$$
=
\left(
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
\right)
\Sigma_j
$$

所以：

$$
\Sigma_j
=
\frac{
\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^T
}{
\sum_{i=1}^m w_j^{(i)}
+
\alpha\sum_{i=1}^{\tilde m}1\{\tilde z^{(i)}=j\}
}
$$

直觉上，$\Sigma_j$ 是第 $j$ 类样本围绕均值 $\mu_j$ 的加权协方差。无标签样本用 responsibility 做软加权，有标签样本用真实标签做硬加权。


## 12. EM 为什么会单调上升

EM 每轮不会降低 likelihood。原因是 E-step 选择 $Q^{(t)}$，让下界在当前参数处贴住原函数：

$$
\ell(\theta^{(t)})=J(Q^{(t)},\theta^{(t)})
$$

M-step 最大化下界，所以：

$$
J(Q^{(t)},\theta^{(t+1)})\ge J(Q^{(t)},\theta^{(t)})
$$

而下界永远不超过真实 likelihood：

$$
\ell(\theta^{(t+1)})\ge J(Q^{(t)},\theta^{(t+1)})
$$

连起来：

$$
\ell(\theta^{(t+1)})
\ge
J(Q^{(t)},\theta^{(t+1)})
\ge
J(Q^{(t)},\theta^{(t)})
=
\ell(\theta^{(t)})
$$

所以：

$$
\ell(\theta^{(t+1)})\ge \ell(\theta^{(t)})
$$

半监督 EM 也是同样逻辑，只是目标函数变成：

$$
\ell_{\mathrm{semi}}(\theta)=\ell_{\mathrm{unsup}}(\theta)+\alpha\ell_{\mathrm{sup}}(\theta)
$$

其中 supervised 部分是普通完整数据 likelihood，在 M-step 中一起被最大化。
