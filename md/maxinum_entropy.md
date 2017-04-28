# 熵
熵(Entropy)起源于热力学，用来衡量一个热力学系统的混乱程度。信息论的创始人香农把这一概念引入信息学中，把随机变量中信息量的数学期望叫做信息熵，他反应了随机变量的不确定程序。

对于随机变量X，其可能的取值为$$\{x_{1}, x_{2}, \cdots, x_{k}\}$$，定义其信息量$$I(x_{i}) = -\log p(X=x_{i})$$，那么根据定义，其信息熵：
```math
H(X) = E[I(x_{i})] = -\sum_{i=1}^{k}p(X=x_{i})\log p(X=x_{i})
```
$$p(X=x_{i})$$表示随机变量$$X$$取值$$x_{i}$$的概率，一般简记为$$p(x_{i})$$。

可以证明$$0 \leqslant H(X) \leqslant \log k$$。
因为$$0 \leqslant p(x_{i}) \leqslant 1$$，所以$$p(x_{i})\log p(x_{i}) \leqslant 0$$，即$$H(X) \geqslant 0$$。

求$$H(X)$$上限等同于求解：
```math
\max\limits_{p} H(X)
```
```math
s.t. \sum_{i=1}^{k}p(x_{i}) = 1
```
构造拉格朗日函数：
```math
L(p) = -\sum_{i=1}^{k}p(x_{i})\log p(x_{i}) + \beta(\sum_{i=1}^{k}p(x_{i}) - 1)
```
求$$L(p)$$对$$p$$的梯度得：
```math
\bigtriangledown_{p} L(p) = \beta -1 - \log p(x_{i})
```
令$$\bigtriangledown_{p} L(p) = 0$$可得：
```math
p(x_{i}) = \exp(\beta - 1)
```
代入到$$\sum_{i=1}^{k}p(x_{i}) = 1$$中得到：
```math
k \cdot \exp(\beta - 1) = 1
```
```math
\beta = 1- \log k
```
代入上式可解得：
```math
p(x_{i}) = \frac{1}{k}
```
```math
\max H(X) =  - k \cdot \frac{1}{k} \cdot \log \frac{1}{k} = \log k
```
即当$$p(x_{1}) = p(x_{2}) = \cdots = p(x_{k}) = \frac{1}{k}$$时，$$X$$的熵$$H(X)$$取得最大值$$\log k$$。

# 条件熵
设$$X \in \{x_{1}, x_{2}, \cdots, x_{m}\}, Y \in \{y_{1}, y_{2}, \cdots, y_{n}\}$$为两个随机变量，在已知$$X$$的条件下，$$Y$$的条件熵为：
```math
H(Y|X) = \sum_{i=1}^{m}p(x_{i})H(Y|X=x_{i}) = -\sum_{i=1}^{m}p(x_{i})\sum_{j=1}^{n}p(y_{j}|x_{i})\log p(y_{j}|x_{i})
```
它表示在已知$$X$$的条件下，$$Y$$的条件概率分布的熵相对于$$X$$的数学期望。

# 最大熵原理
从上面推导$$H(X)$$上限的过程中，可以得出结论，当没有其他的约束条件的情况下，当随机变量所有可能的值的概率相同时，熵最大。最简单的例子就是在抛硬币的时候，我们一般认为正反面的概率各占$$\frac{1}{2}$$，这就是最大熵原理的一个典型应用。上硬币的例子是在理想的环境下进行的实验，现实中，硬币的质量不可能是完全均匀的，而且在上抛和下降的过程中，风向，抛起时的角度都会影响最后的结果。所以在实际问题中，还需要满足已有的约束条件。

我们我们为什么会认为“等概率”是最合理的呢？因为我们在没有其他任何信息的情况下，我们只能认为对于未知的分布模型来说，最随机的，最不确定的才是最客观的，也是风险最小的。改变某个值的概率就意味着在我们没有更多的信息的情况下，我们增加了主观的约束。如果定义“最随机的”呢？在满足约束条件的情况下，显然我们不能再用“等概率”去解决问题，而熵就是表示信息的不确定程度的，而且是一个可优化的数值指标。

最大熵原理就是说在学习概率模型时，在所有满足约束条件的概率模型中，熵最大的模型是最好的模型。

# 最大熵模型
我们将最大熵原理应用在分类问题熵，得到的模型就是最大熵模型。
假设分类模型是一个条件概率分布$$P(Y|X)$$，其中$$X \in \mathcal{X} \subseteq R^{n}, Y \in \mathcal{Y}$$，其中$$\mathcal{X}$$和$$\mathcal{Y}$$分别是输入和输出的集合。这个模型表示对于给定的$$X$$，我们以条件概率$$P(Y|X)$$输出$$Y$$。
给定一个数据集
```math
T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{m}, y_{m}) \}
```
其中$$x_{i} \in \mathcal{X}$$表示输入，$$y_{i} \in \mathcal{Y}$$表示输出，$$m$$为样本个数。我们学习的目标就是利用最大熵原理选择出熵最大的分类模型。

首先，我们把保证模型满足已知的左右约束，那么如何在描述这些约束呢？我们从样本中提取出若干有意义的特征，然后要求这些特征在$$T$$上的经验分布$$\tilde{p}(x,y)$$的数学期望和他们在模型中关于$$p(x,y)$$的数学期望相等。这样，一个特征就对应一个约束。

首先，我们看看如何计算经验分布：
所谓经验分布就是指通过样本数据统计得到的，由此可得：
```math
\tilde{P}(X=x, Y=y) = \frac{v(X =x, Y = y)}{m}
```
```math
\tilde{P}(X=x) = \frac{v(X =x)}{m}
```
其中$$v(X =x, Y = y)$$表示$$(x, y)$$在样本中出现的个数，$$v(X = x)$$表示样本输入中$$x$$出现的个数。

我们假定已经从样本集$$T$$中提取了若干的特征，提取特征不是本文所讨论的内容，一般的，这些特征会用特征函数来表示，特征函数可以是关于样本(x, y)的任意实值函数，这里我们只考虑一种简单的形式：
```math
f(x,y) = \left\{\begin{matrix}
1,  & x, y \space \text{satisfy such a fact}
\\ 0, & otherwise
\end{matrix}\right.
```
特征函数$$f(x, y)$$关于经验分布$$\tilde{p}(x, y)$$的数学期望用$$E_{\tilde{P}}(f)$$表示：
```math
E_{\tilde{P}}(f) = \sum_{x, y}\tilde{P}(x, y)f(x, y)
```

特征函数$$f(x, y)$$关于模型$$p(x, y)$$的数学期望用$$E_{P}(f)$$表示：
```math
E_{P}(f) = \sum_{x, y}p(x, y)f(x, y) = \sum_{x, y}p(x)p(y|x)f(x, y)
```
因为$$p(x)$$是无法计算的，所以这里我们只能用$$\tilde{p}(x)$$来近似$$p(x)$$，即：
```math
E_{P}(f) = \sum_{x, y}\tilde{p}(x)p(y|x)f(x, y)
```
令
```math
C = \{P | E_{P}(f_{i}) = E_{\tilde{P}}(f_{i}), \space i = 1,2, \cdots, n\}
```

这样，我们就可以给出最大熵模型：
对于给定的样本集$$T$$，有特征函数$$f_{i}(x, y), i = 1,2, \cdots, n$$，优化的目标是：
```math
\max \limits_{P \in C} H(p) = -\sum_{x, y}\tilde{p}(x)p(y|x)\log p(y|x)
```
```math
s.t. \quad E_{P}(f_{i}) = E_{\tilde{P}}(f_{i}), \space i = 1,2, \cdots, n
```
```math
\sum_{y}p(y|x) =1
```
为了求解方便，可以将最大值问题转化成最小值问题，即
```math
\min \limits_{P \in C} \sum_{x, y}\tilde{p}(x)p(y|x)\log p(y|x)
```
```math
s.t. \quad E_{P}(f_{i}) = E_{\tilde{P}}(f_{i}), \space i = 1,2, \cdots, n
```
```math
\sum_{y}p(y|x) =1
```
# 模型求解
根据拉格朗日乘数法，引入拉格朗日乘数$$w_{0}, w_{1}, w_{2}, \cdots, w_{n}$$，定义拉格朗日函数$$L(P, w)$$：
```math
L(P, w) = \sum_{x, y}\tilde{p}(x)p(y|x)\log p(y|x) + w_{0}(1 - \sum_{y}p(y|x)) + \sum_{i=1}^{n}w_{i}(E_{\tilde{P}}(f_{i}) - E_{P}(f_{i}))
```
```math
 = \sum_{x, y}\tilde{p}(x)p(y|x)\log p(y|x) + w_{0}(1 - \sum_{x,y}\tilde{p}(x)p(y|x)) + \sum_{i=1}^{n}w_{i}(E_{\tilde{P}}(f_{i}) - \sum_{x, y}\tilde{p}(x)p(y|x)f(x, y))
```
则
```math
\bigtriangledown_{P}L(P, w) = \sum_{x, y}\tilde{p}(x)(1 + \log p(y|x)) - w_{0}\sum_{x,y}\tilde{p}(x) - \sum_{i=1}^{n}w_{i}\sum_{x, y}\tilde{p}(x)f_{i}(x,y)
```
```math
= \sum_{x, y}\tilde{p}(x)(1 + \log p(y|x) - w_{0} - \sum_{i=1}^{n}w_{i}f_{i}(x,y))
```

令$$\bigtriangledown_{P}L(P, w) = 0$$，又因为$$\tilde{p}(x) > 0$$，所以只可能是$$1 + \log p(y|x) - w_{0} - \sum_{i=1}^{n}w_{i}f_{i}(x,y) = 0$$，得：
```math
p(y|x) = \exp(w_{0} - 1) \cdot \exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))
```
代入到$$\sum_{y}p(y|x) =1$$中得：
```math
\exp(w_{0} - 1) = \frac{1}{\sum_{y} \exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))}
```
代入$$p(y|x)$$：
```math
p(y|x) = \frac{\exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))}{\sum_{y} \exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))}
```
设$$Z_{w}(x) = {\sum_{y} \exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))}$$，则
```math
p(y|x) = \frac{1}{Z_{w}(x)}\exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))
```
把$$p(y|x)$$代入到$$E_{P}(f_{i}) = E_{\tilde{P}}(f_{i})$$中，联立这$$n$$个方程可以解得$$w$$。由于这些方程是非线性的，求解的过程是非常复杂的。我们无法直接给出其解的解析形式。

观察$$p(y|x)$$，可以发现形式上和softmax回归的假设函数非常的像。当$$f_{i}(x, y) = v_{i}^{T}x$$时，即$$f_{i}(x, y)$$退化成一个只关于$$x$$的线性函数时，最大熵模型就是softmax回归了。

和softmax回归的求解策略一样，我们可以用最大似然估计来求解参数$$w$$：
```math
L(w) = \prod_{x, y}p(y|x)^{\tilde{p}(x,y)}
```
取对数
```math
l(w) = \sum_{x, y}\tilde{p}(x,y)\log p(y|x)
```
```math
= \sum_{x,y}\tilde{p}(x,y)(\sum_{i=1}^{n}w_{i}f_{i}(x,y) - \log Z_{w}(x))
```
```math
= \sum_{i=1}^{n}w_{i}\sum_{x,y}\tilde{p}(x,y)f_{i}(x,y) - \sum_{x,y}\tilde{p}(x,y)\log Z_{w}(x)
```
```math
= \sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x)\log Z_{w}(x)
```
上面最后一步的推导中，因为$$Z_{w}(x)$$和$$y$$无关，所以类似于多重积分，这里可以把$$y$$直接消掉。

最终的优化目标是：
```math
\max \limits_{w} (\sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x)\log Z_{w}(x))
```
需要说明的是这里用到的是最大似然估计的指数形式。相当于把通常定义的最大似然函数开了$$m$$次方，这是为了引出$$\tilde{p}(x,y)$$，本质上并没有区别。

再换一个角度，注意到，我们优化的目标$$-H(X)$$是一个凸函数，而且$$h(x) = E_{\tilde{P}}(f_{i}) - E_{P}(f_{i}) $$是一组仿射函数。这样，很容易联系到拉格朗日函数的对偶问题和KKT条件。

我们优化的原始问题等价于
```math
\min \limits_{P \in C} \max \limits_{w} L(P, w)
```
对偶问题是
```math
\max \limits_{w} \min \limits_{P \in C} L(P, w)
```

因为满足KKT条件，所以可以通过求解对偶问题来求解原始问题。

首先把$$w$$看成常量来求解$$\min \limits_{P \in C} L(P, w)$$，令$$\bigtriangledown_{P}L(P, w) = 0$$可得(上面已经计算过)：
```math
p^{*}(y|x) = \frac{1}{Z_{w}(x)}\exp(\sum_{i=1}^{n}w_{i}f_{i}(x,y))
```
优化目标变成$$\max \limits_{w}L(P^{*}, w)$$
把$$p^{*}(y|x)$$代入$$L(P, w)$$：
```math
L(P^{*}, w) = \sum_{x, y}\tilde{p}(x)p^{*}(y|x)\log p^{*}(y|x) + w_{0}(1 - \sum_{y}p^{*}(y|x)) + \sum_{i=1}^{n}w_{i}(E_{\tilde{P}}(f_{i}) - E_{P}(f_{i}))
```
```math
= \sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) - \sum_{x, y}\tilde{p}(x)p^{*}(y|x)(\log p^{*}(y|x) - \sum_{i=1}^{n}w_{i}f_{i}(x, y))
```
```math
= \sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) + \sum_{x}\tilde{p}(x)(\sum_{y}p^{*}(y|x))(\sum_{i=1}^{n}w_{i}f_{i}(x, y) - \log Z_{w}(x) - \sum_{i=1}^{n}w_{i}f_{i}(x, y))
```
```math
= \sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x)\log Z_{w}(x)
```
推导过程中，两次用到了$$\sum_{y}p^{*}(y|x) =1$$。
最后的优化目标就是
```math
\max \limits_{w} (\sum_{i=1}^{n}w_{i}E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x)\log Z_{w}(x))
```
可以看到，利用拉格朗日函数推导出的结果和从极大似然估计得出的结果是相同的。

最后，我们需要用最优化算法来解决这个凸优化问题。
# 最优化算法
最容易想到的算法就是梯度上升法。很容易计算最大对数似然函数$$l(w)$$对$$w$$的梯度：
```math
\bigtriangledown_{w}l(w) = (\frac{\partial l(w)}{\partial w_{1}}, \frac{\partial l(w)}{\partial w_{2}}, \cdots, \frac{\partial l(w)}{\partial w_{n}})^{T}
```
```math
\frac{\partial l(w)}{\partial w_{i}} = E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x) \cdot \frac{1}{Z_{w}(x)} \cdot \frac{\partial Z_{w}(x)}{\partial w_{i}}
```
```math
= E_{\tilde{P}}(f_{i}) - \sum_{x}\tilde{p}(x)\cdot \frac{\sum_{y}\exp(\sum_{i=1}^{n}w_{i}f_{i}(x, y)) \cdot f_{i}(x, y)}{Z_{w}(x)}
```
```math
= E_{\tilde{P}}(f_{i}) - \sum_{x, y}\tilde{p}(x) \cdot  \frac{\exp(\sum_{i=1}^{n}w_{i}f_{i}(x, y))}{Z_{w}(x)} \cdot f_{i}(x, y)
```
```math
= E_{\tilde{P}}(f_{i}) - \sum_{x, y}\tilde{p}(x)p(y|x)f_{i}(x, y)
```

此外，还有一种专为最大熵模型设计的优化算法，叫做GIS(Generalized Iterative Scaling)算法:
1. 初始化参数，令 $$w_{i} = 0$$
2. 计算$$E_{\tilde{P}}(f_{i})$$
3. 计算$$E_{p}(f_{i})$$
4. 更新参数：$$w_{i} = w_{i} + \eta\log \frac{E_{\tilde{P}}(f_{i})}{E_{p}(f_{i})} $$
5. 若$$|\log \frac{E_{\tilde{P}}(f_{i})}{E_{p}(f_{i})}| \lt \epsilon$$，则停止迭代，否则转到第3步

其中的$$\eta$$类似梯度下降法中的学习率，一般可设为$$\frac{1}{\sum_{i=1}^{n}f_{i}(x,y)}$$。
GIS算法的问题是每次迭代需要的时间都很长，而且需要很多次迭代才能收敛，后来有人提出了改进的IIS(Improved Iterative Scaling)算法，使得最大熵模型变得实用。
1. 初始化参数，令 $$w_{i} = 0$$
2. 令$$\delta_{i}$$是方程
```math
\sum_{x, y}\tilde{p}(x)p(y|x)f_{i}(x,y)\exp(\delta_{i}\sum_{i=1}^{n}f_{i}(x,y)) = E_{\tilde{P}}(f_{i})
```
的解。
3. 更新参数：$$w_{i} = w_{i} + \delta_{i}$$
4. 若$$|\delta_{i}| \lt \epsilon$$，则停止迭代，否则转到第2步

可以看出，GIS算法和IIS算法的不同就是每个迭代参数的变化量是不同的。
IIS算法的关键就是第2步中的方程的求解，若$$\sum_{i=1}^{n}f_{i}(x,y)=C$$是一个常量，可以直接求解得:
```math
\delta_{i} = \frac{1}{C}\log \frac{E_{\tilde{P}}(f_{i})}{E_{p}(f_{i})}
```
这和GIS算法是一致的。

若$$\sum_{i=1}^{n}f_{i}(x,y)$$不是一个常量，则需要我们通过数值方法来求解，最简单的就是牛顿法。

最后，我们再简单讨论以下IIS算法的原理。
IIS算法的思路是：既然优化的目标是对数似然函数，那么能不能找到一个参数更新的方法，可以使得每一次迭代参数更新后，对数似然函数的值都会变得更大。
我们设每次迭代参数$$w$$的变化量是$$\delta$$，则：
```math
L(w + \delta) - L(w) = \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) - \sum_{x}\tilde{p}(x)\log \frac{Z_{w+\delta}(x)}{Z_{w}(x)}
```
```math
\geqslant \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\frac{Z_{w+\delta}(x)}{Z_{w}(x)}
```
```math
= \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)\exp(\sum_{i=1}^{n}\delta_{i}f_{i}(x, y))
```
上面从等式变成不等式的过程用了$$-\log x \geqslant 1 - x , x \gt 0$$
令
```math
A(\delta|w) = \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)\exp(\sum_{i=1}^{n}\delta_{i}f_{i}(x, y))
```
则
```math
L(w + \delta) - L(w) \geqslant A(\delta|w)
```

令$$f^{*}(x, y) = \sum_{i=1}^{n}f_{i}(x, y)$$
```math
A(\delta|w) = \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)\exp(f^{*}(x, y)\sum_{i=1}^{n}\delta_{i}\frac{f_{i}(x, y)}{f^{*}(x, y)})
```
因为$$\frac{f_{i}(x, y)}{f^{*}(x, y)} \geqslant 0$$，$$\sum_{i=1}^{n}\frac{f_{i}(x, y)}{f^{*}(x, y)} = 1$$，且指数函数是凸函数，若把$$\frac{f_{i}(x, y)}{f^{*}(x, y)}$$看成是某个概率，根据Jensen不等式(见EM算法中的介绍)有：
```math
\exp(f^{*}(x, y)\sum_{i=1}^{n}\delta_{i}\frac{f_{i}(x, y)}{f^{*}(x, y)}) \leqslant \sum_{i=1}^{n}\frac{f_{i}(x, y)}{f^{*}(x, y)}\exp(\delta_{i}f^{*}(x, y))
```
所以有:
```math
A(\delta|w) \geqslant \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)\sum_{i=1}^{n}\frac{f_{i}(x, y)}{f^{*}(x, y)}\exp(\delta_{i}f^{*}(x, y))
```
令
```math
B(\delta|w) = \sum_{x, y}\tilde{p}(x, y)\sum_{i=1}^{n}\delta_{i}f_{i}(x, y) + 1 - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)\sum_{i=1}^{n}\frac{f_{i}(x, y)}{f^{*}(x, y)}\exp(\delta_{i}f^{*}(x, y))
```
则有
```math
L(w + \delta) - L(w) \geqslant B(\delta|w)
```
再求$$B(\delta|w)$$对$$\delta_{i}$$的偏导，令偏导为0，求出使$$B(\delta|w)$$取得最大的$$\delta$$.
```math
\frac{\partial B(\delta|w)}{\partial \delta_{i}} = \sum_{x, y}\tilde{p}(x, y)f_{i}(x, y) - \sum_{x}\tilde{p}(x)\sum_{y}p(y|x)f_{i}(x, y)\exp(\delta_{i}f^{*}(x, y))
```
```math
= E_{\tilde{P}}(f_{i}) - \sum_{x, y}\tilde{p}(x)p(y|x)f_{i}(x,y)\exp(\delta_{i}f^{*}(x, y)) = 0
```
这就是IIS算法第2步的方程。
