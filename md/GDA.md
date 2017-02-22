高斯辨别分析是一种生成学习算法。在该模型中，假定对于给定的目标y的zhi值，x服从多元高斯分布。根据训练集合中的样本数据来拟合参数。再将新样本的数据带入到模型中分别计算样本属于各个分类的概率，并概率最大的作为其所属的类别。

# 多元高斯分布

多元高斯分布的概率密度函数为：
```math
p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{\frac{\pi}{2}}|\Sigma|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))
```

其中，$$\mu$$为多元高斯分布的期望，$$\Sigma$$为数据集中X的协方差，$$|\Sigma|$$表示为协方差的行列式，是一个标量。若X为$$m\times n$$的矩阵，则$$\mu \in R^{n}, \Sigma \in R^{n \times n}$$


## 协方差矩阵
协方差是一个统计学概率，来用衡量两个随机变量的总体误差。

期望值分别为$$E[X]$$与$$E[Y]$$的两个实随机变量X与Y之间的协方差$$Cov(X,Y)$$定义为：

```math
Cov(X, Y) = E[(X-E(X))(Y-E(Y))]

=E[XY] - 2E[X]E[Y] + E[X]E[Y]

=E[XY] - E[X]E[Y]
```
若X与Y是独立统计的，那么有$$E[XY] = E[X]E[Y]$$,则二者的协方差wei为0，但是反过来却不一定成立


协方差矩阵则是各个向量元素之间的协方差组成的矩阵，记为$$\Sigma$$,

```math
\Sigma(X) = E[(X - E(X))(X - E(X))^{T}]
```
```math
= \begin{bmatrix}
 E((X_{1}-\mu_{1})(X_{1}-\mu_{1})) & E((X_{1}-\mu_{1})(X_{2}-\mu_{2}))   &
 \cdots &
 E((X_{1}-\mu_{1})(X_{n}-\mu_{n})) \\
 E((X_{2}-\mu_{2})(X_{1}-\mu_{1}))  & E((X_{2}-\mu_{2})(X_{2}-\mu_{2}))  &
 \cdots &
 E((X_{2}-\mu_{2})(X_{n}-\mu_{n})) \\  
 \vdots & \vdots & \ddots & \vdots \\
 E((X_{n}-\mu_{n})(X_{1}-\mu_{1}))  & E((X_{n}-\mu_{n})(X_{2}-\mu_{2}))  &
 \cdots&
 E((X_{n}-\mu_{n})(X_{n}-\mu_{n}))
\end{bmatrix}
```

由此可见，协方差矩阵具有对称性和正定性，在主成分分析(Principal Components Analysis)的过程中，也需要用协方差矩阵来去除数据的相关性。

# 高斯辨别分析模型

假设给定的数据集是一个二分类问题，则可以认为目标y是服从伯努利分布的

```math
y \sim Bernoulli(\phi)
```
```math
x|y=0 \sim N(\mu_{0}, \Sigma_{0})
```
```math
x|y=1 \sim N(\mu_{1}, \Sigma_{1})
```

则他们的概率密度函数是：
```math
p(y) = \phi^{y}(1-\phi)^{1-y}
```
```math
p(x|y=0; \mu_{0}, \Sigma_{0}) = \frac{1}{(2\pi)^{\frac{\pi}{2}}|\Sigma_{0}|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-\mu_{0})^{T}\Sigma_{0}^{-1}(x-\mu_{0}))
```
```math
p(x|y=1; \mu_{1}, \Sigma_{1}) = \frac{1}{(2\pi)^{\frac{\pi}{2}}|\Sigma_{1}|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-\mu_{1})^{T}\Sigma_{1}^{-1}(x-\mu_{1}))
```
