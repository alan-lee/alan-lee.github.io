# 线性回归
有样本集$$(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})$$，其中$$x^{(i)} \in R^{n}, y^{(i)} \in R, i = 1, 2, \cdots, m$$,
我们假设其输出和输入是线性关系，即假设函数$$h_{\theta, b}(x) = \theta^{T}x + b$$，其中$$\theta$$为特征值的权重，$$b$$是偏差值。为了使计算更加方便，我们稍微改变一下样本集，在每个输入$$x^{(i)}, i= 1, 2, \cdots, m$$的最前面插入一个常数$$1$$，这样新的$$ x^{(i)} \in R^{n + 1}$$。
我们把假设函数改写成
```math
h_{\theta} = \theta^{T}x
```
其中$$\theta = (\theta_{0}, \theta_{1}, \theta_{2}, \cdots, \theta_{n})^{T},\space \theta_{0} = b$$

上述模型通过对样本集的学习，可以得到一组最优化的参数$$\theta^{*}$$，然后就可以用$$\theta^{*}$$来估算新的样本的输出值。

由最小二乘法的思想，我们求解参数$$\theta$$的策略就是要使训练样本集的误差最小，定义损失函数：
```math
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(y^{(i)} - h_{\theta}(x^{(i)}))^{2}
```
那么$$\theta^{*} = \arg\min \limits_{\theta} J(\theta)$$

这一损失函数也称为平方损失函数

令$$X = (x^{(1)}, x^{(2)}, \cdots, x^{(m)})^{T}, y = (y^{(1)}, y^{(2)}, \cdots, y^{(m)})^{T}$$，则$$X \in R^{m \times n = 1}, y \in R^{m}$$

若满足$$X^{T}X$$是一个非奇异矩阵(满秩矩阵)，则
```math
\theta^{*} = (X^{T}X)^{-1}X^{T}y
```

在求出$$\theta^{*}$$后，对于新的样本，我们在新样本的第一个特征前加上常量$$1$$，然后可以用$$h_{\theta} = \theta^{*T}x$$来做预测。

我们假定样本误差$$\varepsilon_{i} = |y^{(i)} - \theta^{*T}x^{(i)}|$$是符合正态分布的，即
```math
\varepsilon_{i} \sim N(0, \sigma^{2})
```

注意这里的每个样本的误差都独立的服从一个期望和方差都相同的正态分布。这是一个观察的得出的结果。

可以得出
```math
y^{(i)} \sim N(\theta^{*T}x^{(i)}, \sigma^{2})
```
那么最大似然估计函数为：
```math
L(\theta) = \prod_{i=1}^{m}\frac{1}{2\pi\sigma}\exp(-\frac{(y^{(i)} - \theta^{T}x^{(i)})^{2}}{2\sigma^{2}})
```

取对数
```math
l(\theta) = m\log\frac{1}{2\pi\sigma} - \frac{1}{2\sigma^{2}}\sum_{i=1}^{m}(y^{(i)} - \theta^{T}x^{(i)})^{2}
```

对比$$l(\theta)$$和$$J(\theta)$$可以发现$$\max \limits_{\theta}l(\theta)$$和$$\min \limits_{\theta}J(\theta)$$是等价的。
这也是我们选用平方损失函数的原因。


# Logistic回归
线性回归是用来预测目标的值，例如：房价等。其结果集一般都是连续分布的，对于结果集是离散的标量，例如分类问题，线性回归是解决不了的。

对于分类问题，我们可以从概率的角度去思考：先计算出样本属于每个分类的概率，然后再选概率最大的分类作为预测的结果。

Logistic回归是用来处理二分类的问题，样本的结果$$y \in \{0, 1\}$$，因此，我们可以把每个样本的结果看成是独立服从于伯努利分布的。
设
```math
p(y = 1 | x) = \phi
```
则：
```math
y^{(i)} \sim Bernoulli(\phi)
```
事件的几率(odds)：一个事件发生的几率是指该事件发生的概率与不发生的概率的比值。如果一个事件发生的概率是$$p$$，那么该事件的几率是$$\frac{p}{1 - p}$$。对几率取对数就是对数几率，记为$$logit(p) = \log \frac{p}{1 - p}$$

Logistic回归的模型就是：假设“样本结果为1”这一事件的对数几率和样本特征之间是线性的关系，即：
```math
\log \frac{\phi}{1 - \phi} = \theta^{T}x
```
可得：
```math
\phi = \frac{\exp(\theta^{T}x)}{1 + \exp(\theta^{T}x)} = \frac{1}{1 + \exp(-\theta^{T}x)}
```

从上面的概率中，我们引出Sigmoid函数：
```math
g(x) = \frac{1}{1 + \exp(-x)}
```
下图是Sigmoid函数在坐标轴上的图像，形状类似于S形的曲线，其值在区间$$(0, 1)$$内，和纵轴的交点是$$0.5$$，函数值的变化速度在靠近纵轴时加快，远离纵轴时变缓。
![image](http://oirw7ycq0.bkt.clouddn.com/sigmoid.png)

当$$g(\theta^{T}x) \geqslant 0.5$$时，$$p(y=1|x;\theta) \geqslant 0.5$$,可以认为样本结果$$y = 1$$，当$$g(\theta^{T}x) \lt 0.5$$时，认为样本结果$$y = 0$$。

根据以上的模型，我们得出Logistic回归的假设函数就可以写成
```math
h_{\theta}(x) = p(y = 1| x; \theta) = g(\theta^{T}x) = \frac{1}{1 + \exp(-\theta^{T}x)}
```
对于概率最大化的问题，我们一般是用最大似然估计函数来求解的。
由伯努利分布的概率知：
```math
p(y | x; \theta) = \phi^{y}(1 - \phi)^{1 -y} =
```
最大似然估计函数为
```math
L(\theta) = \prod_{i=1}^{m}h_{\theta}(x^{(i)})^{y^{(i)}}(1 - h_{\theta}(x^{(i)}))^{1 -y^{(i)}}
```
最大对数似然估计函数为
```math
l(\theta) = \sum_{i=1}^{m}(y^{(i)}\log h_{\theta}(x^{(i)}) + (1 - y^{(i)})\log(1 - h_{\theta}(x^{(i)})))
```
我们可以用梯度上升或是拟牛顿法来求解$$\theta$$的极大似然估计值$$\theta^{*}$$。
预测新样本分类是，用$$\theta^{*}$$来计算新样本在各个分类的概率即可。

Logistics回归的损失函数其实也是由最大似然估计函数得到的：
```math
J(\theta) = -\frac{1}{m}l(\theta) = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log h_{\theta}(x^{(i)}) + (1 - y^{(i)})\log(1 - h_{\theta}(x^{(i)})))
```
这一损失函数也称为对数损失函数。

最后，我们再直观的分析一下，为什么我们要用对数比率来建立Logistics回归的模型。

如下图所示，对于可以用Logistics回归模型是用来学习的分类问题，我们可以在其中画出一条直线，把正负样本分开。这条直线的方程就是$$\theta^{*T}x$$，其中$$\theta^{*}$$就是我们上面求出的参数的最大似然估计值。
设事件发生的概率为$$p$$，则其比率为$$\frac{p}{1-p}$$，当$$\frac{p}{1-p} \gt 1$$时，事件发生的概率大，否则， 事件发生的概率小。在取对数后，则有当$$logit(p) \gt 0$$时，事件发生的概率大，否则， 事件发生的概率小，如果$$logit(p)$$在某个空间可以用一条直线来表示，则在直线之上的点就是正样本，在直线之下的点就是负样本。Logistics回归的本质就是通过学习来找到这样的一条直线。

![image](http://oirw7ycq0.bkt.clouddn.com/logistic_regression.png)

上图是在二维空间的情况，这一结论也可以推广多维空间。
所以说Logistic回归也是线性回归的一种变形。

# 广义线性模型
线性回归的样本结果服从正态分布，Logistic回归的样本结果服从伯努利分布，正态分布和伯努利分布都是指数分布族的一种。指数分布族的概率密度函数可以写成下面的形式：
```math
p(y;\eta) = b(y)\exp(\eta^{T}T(y) - a(\eta))
```
其中$$\eta$$称为自然系数，和输入变量$$x$$是线性相关的，即$$\eta = \theta^{T}x$$。$$T(y)$$叫做充分统计量，我们要预测的结果，就是$$T(y)$$的期望值。

以Logistic回归来举例说明。
```math
p(y;\phi) = \phi^{y}(1 - \phi)^{1 - y}
```
```math
= \exp(y\log\phi + (1 - y)\log(1 - \phi))
```
```math
=\exp(y\log(\frac{\phi}{1 - \phi}) + \log(1 - \phi))
```
则有：
```math
b(y) = 1
```
```math
T(y) = y
```
```math
\eta = \log(\frac{\phi}{1 - \phi})
```
```math
a(\eta) = -\log(1 - \phi)
```
由
```math
\eta = \theta^{T}x = \log(\frac{\phi}{1 - \phi})
```
得：
```math
\phi = \frac{1}{1 + \exp(-\theta^{T}x)}
```
假设函数：
```math
h_{\theta}(x) = E[y|x;\theta] = \phi * 1 + (1 - \phi) * 0  = \phi
```
```math
= \frac{1}{1 + \exp(-\theta^{T}x)}
```

这和我们建立的Logistic模型时得出的假设函数是一样的。

除了正态分布和伯努利分布，还有很多分布都属于指数分布族，比如泊松分布，Gamma分布，指数分布，多元高斯分布，Beta分布，Dirichlet分布，Wishart分布等等。根据这些分布的概率密度函数可以建立相应的模型，这些都是广义线性模型的一个实例。
