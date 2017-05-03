# Boosting算法
Boosting字面的意思就是提升，Boosting算法可以将多个弱学习器进行结和，以提升其性能。所谓的弱学习器(又叫做基本学习器)是指泛化能力略优于随机猜测的学习器，例如在二分类问题上精度略高于50%的分类器，和其对应的算法正确率很高的就是强学习器。这就像人们在做重大决定时，一般都会考虑吸取多个人的建议。

Boosting算法的机制大致相同：首先从初始的样本集分布中训练出一个基本学习器，再根据基本学习器的表现对样本集的分布进行调整，使得之前的基本学习器判断错误的样本在后续受到更多的关注，直到基本学习器的数量达到事先指定的值$$T$$，最后将这$$T$$个基本学习器组合成一个强学习器。这样，在设计Boosting算法的时候，我们必须解决两个问题：
1. 如何在每一轮训练基本学习器后，根据训练结果改变样本的概率分布
2. 如何将多个弱学习器组合成一个强学习器。

Boosting算法中，最具有代表性的就是AdaBoosting(Adaptive Boosting)算法。
# AdaBoost算法介绍
AdaBoosting算法是由Yoav Freund和Robert Schapire在1995年提出，主要用于解决分类问题，它的主要思想是：
1. 分而治之，对于这一轮被基本分类器分错的样本，分布概率会提高，使其得到更多的关注，而被正确分类的样本的分布概率则会降低。
2. 加权多数表决，在最终组成强分类器时，加大错误率小的基本分类器的权重，使其在最终表决中起较大的作用，同时减少错误率较大的基本分类器的权重。

AdaBoosting 算法的流程如下：
有数据集$$D = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{m}, y_{m})\}$$，其中，$$x_{i} \in \mathcal{X} \subseteq R^{n}, y_{i} \in \mathcal{Y} = \{+1, -1\}$$，用$$G_{t}(x)$$来表示第$$t$$轮训练得到的弱分类器，共需$$T$$个弱分类器。
1.初始化样本分布：
```math
D_{1} = (w_{11}, \cdots, w_{1i}, \cdots, w_{1m}), \quad w_{1i} = \frac{1}{m}, i = 1, 2, \cdots, m
```

2.对于$$t = 1, 2, \cdots, T$$
2.1使用具有概率分布$$D_{t}$$的样本数据学习，得到弱分类器：
```math
G_{t}(x):\mathcal{X} \rightarrow \{+1, -1\}
```
2.2计算$$G_{t}(x)$$在样本数据集上的分类误差率：
```math
e_{t} = P(G_{t}(x_{i}) \neq y_{i}) = \sum_{i=1}^{m}w_{ti}I(G_{t}(x_{i}) \neq y_{i})
```
2.3计算$$G_{t}(x)$$的加权系数
```math
\alpha_{t} = \frac{1}{2}\log \frac{1 - e_{t}}{e_{t}}
```
这里的对数是自然对数。
2.4更新样本数据的概率分布
```math
D_{t+1} = (w_{t+1,1}, \cdots, w_{t+1,i}, \cdots, w_{t+1,m})
```
```math
w_{t+1,i} = \frac{w_{ti}}{Z_{m}}\exp(-\alpha_{t}y_{i}G_{t}(x_{i})), i = 1, 2, \cdots, m
```
这里的$$Z_{t}$$是规范化因子，以确保$$D_{t+1}$$是一个分布。
```math
Z_{t} = \sum_{i=1}^{m}w_{ti}\exp(-\alpha_{t}y_{i}G_{t}(x_{i}))
```
3.得到最终分类器
```math
G(x) = \text{sign}(\sum_{t=1}^{T}\alpha_{t}G_{t}(x))
```
对上面AdaBoost算法的流程做一些分析：
1.错误率：$$e_{t}$$可以理解为$$G_{t}(x)$$在样本分布$$D_{t}$$上对分类错误次数的数学期望。我们可以把$$G_{t}(x_{i})$$的结果是否错误记为$$I(G_{t}(x_{i}) \neq y_{i})$$，所有有：
```math
e_{t} = E_{D_{t}}[I(G_{t}(x_{i}) \neq y_{i})] = \sum_{i=1}^{m}w_{ti}I(G_{t}(x_{i}) \neq y_{i})
```
2.加权系数：$$\alpha_{t}$$表示弱分类器$$G_{t}(x)$$作用于最终分类器的权重。由$$\alpha_{t}$$的定义可知：当$$e_{t} \leqslant \frac{1}{2}$$时，$$\alpha_{t} \geqslant 0$$，并且$$\alpha_{t}$$随着$$e_{t}$$的减小而增大，即误差越小，弱分类器在最终分类器中的作用越大。
3.更新样本分布概率：首先我们加入规范化因子是为了保证$$\sum_{i=1}^{m}w_{ti} = 1$$，即$$D_{t}$$是一个合法的分布。当$$G_{t}(x)$$对某个样本$$(x_{i}, y_{i})$$分类错误时，$$y_{i}G_{t}(x) = -1$$，这时$$\exp(-\alpha_{t}y_{i}G_{t}(x_{i})) = \exp(\alpha_{t})$$，当$$\alpha_{t} \geqslant 0$$时，$$exp(\alpha_{t}) \geqslant 1$$，即其分布概率变大，同理，当分类正确时，其概率变小。
4.在上面的解释中，我们只讨论了$$\alpha_{t} \geqslant 0$$的情况，即$$e_{t} \leqslant \frac{1}{2}$$，弱分类器的错误率要比$$\frac{1}{2}$$大，否则，算法会发散。
# AdaBoost算法推导
Adaboost算法可以认为是模型是加法模型，损失函数为指数损失函数，学习算法是前向分布算法的二分类学习方法。

加法模型：
```math
f(x) = \sum_{t=1}^{T}\beta_{t}b(x;\gamma_{t})
```
其中，$$b(x;\gamma_{t})$$为基函数，$$\gamma_{t}$$为基函数的参数，$$\beta_{t}$$为基函数的系统。
在给定的样本数据集和损失函数$$L(y, f(x))$$的条件下，加法模型的学习过程变成经验风险最小化问题：
```math
\min \limits_{\beta, \gamma}\sum_{i=1}^{m}L(y_{i}, \sum_{t=1}^{T}\beta_{t}b(x_{i};\gamma_{t}))
```
对于AdaBoosting算法，则有：
```math
f(x) = \sum_{t=1}^{T}\alpha_{t}G_{t}(x)
```
```math
L(y, f(x)) = \exp(-yf(x))
```
AdaBoost算法的最优化问题就是
```math
\min \limits_{\alpha} \sum_{i=1}^{m} \exp(-y_{i}\sum_{t=1}^{T}\alpha_{t}G_{t}(x_{i}))
```

这是一个很复杂的优化问题，这里我们采用向前分布算法来求解：
1.初始化$$f_{0} = 0$$
2.对于$$t = 1, 2, \cdots, T$$
2.1最小化损失函数
```math
(\beta_{t}, \gamma_{t}) = \arg\min\limits_{\beta, \gamma} \sum_{i=1}^{m}L(y_{i}, f_{t-1}(x_{i}) + \beta b(x_{i}; \gamma))
```
得到参数$$\beta_{t}, \gamma_{t}$$
2.2更新
```math
f_t(x) = f_{t-1}(x) + \beta_{t} b(x; \gamma_{t})
```
3.得到加法模型
```math
f(x) = f_{T}(x) = \sum_{t=1}^{T}\beta_{t}b(x;\gamma_{t})
```

向前分布算法的思路很简单，从后向前，每一步只学习一个基函数和其加权系数，逐步接近优化目标。

对于AdaBoost算法，
```math
f_{t}(x) = f_{t-1}(x) + \alpha_{t}G_{t}(x)
```
```math
L_{t}(\alpha_{t}, G_{t}(x)) = \sum_{i=1}^{m}L(y_{i}, f_t(x_{i})) = \sum_{i=1}^{m}\exp(-y_{i}(f_{t-1}(x_{i}) + \alpha_{t}G_{t}(x_{i})))
```
```math
=\sum_{i=1}^{m}\exp(-y_{i}f_{t-1}(x_{i})) \cdot \exp(-y_{i}\alpha_{t}G_{t}(x_{i}))
```
```math
=\sum_{y_{i} \neq G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(a_{t}) + \sum_{y_{i} = G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(-a_{t})
```
求$$L_{t}(\alpha_{t}, G_{t}(x))$$对$$\alpha_{t}$$的偏导：
```math
\frac{\partial L_{t}(\alpha_{t}, G_{t}(x))}{\partial \alpha_{t}} = \sum_{y_{i} \neq G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(a_{t}) - \sum_{y_{i} = G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(-a_{t})
```
令偏导数为0可得：
```math
\sum_{y_{i} \neq G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(a_{t}) = \sum_{y_{i} = G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))\exp(-a_{t})
```

```math
\exp(2\alpha_{t}) = \frac{\sum_{y_{i} = G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))}{\sum_{y_{i} \neq G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))}
```
解得：
```math
\alpha_{t} = \frac{1}{2} \log \frac{\sum_{y_{i} = G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))}{\sum_{y_{i} \neq G_{t}(x_{i})}\exp(-y_{i}f_{t-1}(x_{i}))}
```
令$$\bar{w}_{ti} = \exp(-y_{i}f_{t-1}(x_{i}))$$，展开可得：
```math
\bar{w}_{ti} = \exp(-y_{i}f_{t-1}(x_{i}))
```
```math
=\exp(-y_{i}f_{t-2}(x_{i})) \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
=\bar{w}_{t-1,i}.\exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
注意观察，$$\bar{w}_{ti}$$和$$w_{ti}$$比，缺少了规范化因子$$Z_{m}$$，
根据定义$$f_{0}(x) = 0$$，所以$$\bar{w}_{1i} = 1$$，在对$$\bar{w}_{1i}$$进行规范化后，可得
```math
w_{1i} = \frac{\bar{w}_{1i}}{\sum_{j=1}^{m}\bar{w}_{1j}} = \frac{1}{m}
```
同理，
```math
w_{ti} = \frac{\bar{w}_{ti}}{\sum_{j=1}^{m}\bar{w}_{tj}}
```
则代入到$$\alpha_{t}$$中，有：
```math
\alpha_{t} = \frac{1}{2} \log \frac{\sum_{y_{i} = G_{t}(x_{i})}w_{ti}}{\sum_{y_{i} \neq G_{t}(x_{i})}w_{ti}}
```
```math
=\frac{1}{2} \log \frac{\sum_{i=1}^{m}w_{ti} I(y_{i} = G_{t}(x_{i}))}{\sum_{i=1}^{m}w_{ti} I(y_{i} \neq G_{t}(x_{i}))}
```
令错误率$$e_{t} = \sum_{i=1}^{m}w_{ti} I(y_{i} \neq G_{t}(x_{i}))$$，则
```math
\alpha_{t} = \frac{1}{2} \log \frac{1 - e_{t}}{e_{t}}
```
这样，我们就推导出了$$\alpha_{t}$$，形式上和我们给出的结果一样。

AdaBoost算法的关键点除了$$\alpha_{t}$$，还有$$w_{ti}$$，这里给出的$$w_{ti}$$的计算方式和上面还略有不同。下面继续推导：
```math
w_{ti} = \frac{\bar{w}_{ti}}{\sum_{j=1}^{m}\bar{w}_{tj}} = \frac{\bar{w}_{t-1,i}}{\sum_{j=1}^{m}\bar{w}_{tj}} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
= \frac{w_{t-1,i} \cdot \sum_{l=1}^{m}\bar{w}_{t-1,l}}{\sum_{j=1}^{m}\bar{w}_{tj}} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
= \frac{\sum_{l=1}^{m}\bar{w}_{t-1,l}}{\sum_{j=1}^{m}\bar{w}_{tj}} \cdot w_{t-1,i} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
再看
```math
\frac{\sum_{j=1}^{m}\bar{w}_{t,j}}{\sum_{l=1}^{m}\bar{w}_{t-1,l}} = \sum_{j=1}^{m} \frac{\bar{w}_{t,j}}{\sum_{l=1}^{m}\bar{w}_{t-1,l}}
```
```math
=\sum_{j=1}^{m} \frac{\bar{w}_{t-1,j}}{\sum_{l=1}^{m}\bar{w}_{t-1,l}} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
=\sum_{j=1}^{m}w_{t-1, j} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
=Z_{m-1}
```
代入得：
```math
w_{ti} = \frac{\sum_{l=1}^{m}\bar{w}_{t-1,l}}{\sum_{j=1}^{m}\bar{w}_{tj}} \cdot w_{t-1,i} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
```math
= \frac{w_{t-1,i}}{Z_{m-1}} \cdot \exp(-y_{i}\alpha_{t-1}G_{t-1}(x_{i}))
```
 这就和AdaBoost算法中的形式一样了。

最后再讨论关于指数损失函数的话题。
首先把损失函数对$$f(x)$$求偏导
```math
\frac{\partial L(y, f(x))}{\partial f(x)} = -\exp(-f(x))P(y = 1 | x) + \exp(f(x))P(y = -1 | x)
```
令其偏导数为0可得：
```math
f(x) = \frac{1}{2}\log \frac{P(y = 1 | x)}{P(y = -1 | x)}
```
则
```math
G(x) = \text{sign}(f(x)) = \text{sign}(\frac{1}{2}\log \frac{P(y = 1 | x)}{P(y = -1 | x)})
```
```math
= \left\{\begin{matrix}
1,  & P(y = 1 | x) \gt P(y = -1 | x)
\\ -1, & P(y = 1 | x) \lt P(y = -1 | x)
\end{matrix}\right.
```

这表明$$G(x)$$完全是依据先验经验来判断的，即达到了贝叶斯最优错误率，也就是说当损失函数最小时，分类的错误率也是最小的。这表明用指数损失函数是分类任务原本的0/1损失函数的一致替代损失函数。而且，指数函数具有更好的数学性质，例如在实数域上是连续$$n$$阶可导的。下图展示了指数函数和0/1函数的对比，直观上看，用指数函数代替0/1函数做分类也是可行的。
![image](http://oirw7ycq0.bkt.clouddn.com/express.jpg)

# AdaBoost算法误差分析
我们定义AdaBoost算法的误差为：
```math
\varepsilon = \frac{1}{m}\sum_{i=1}^{m}I(G(x_{i}) \neq y_{i})
```
因为当$$G(x_{i}) = y_{i}$$时，$$\exp(-y_{i}f(x_{i})) \gt 0 =  I(G(x_{i}) \neq y_{i})$$，当$$G(x_{i}) \neq y_{i}$$时，$$\exp(-y_{i}f(x_{i})) \geqslant 1 =  I(G(x_{i}) \neq y_{i})$$，所以有：
```math
\varepsilon \leqslant \frac{1}{m}\sum_{i=1}^{m}\exp(-y_{i}f(x_{i}))
```
又有
```math
\frac{1}{m}\sum_{i=1}^{m}\exp(-y_{i}f(x_{i})) = \frac{1}{m}\sum_{i=1}^{m}\exp(-\sum_{t=1}^{T}\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=\sum_{i=1}^{m}\frac{1}{m}\exp(-\sum_{t=1}^{T}\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=\sum_{i=1}^{m}w_{1i}\prod_{t=1}^{T}\exp(\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=\sum_{i=1}^{m}(w_{1i} \cdot \exp(\alpha_{1}y_{i}G_{1}(x_{i}))\prod_{t=2}^{T}\exp(\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=\sum_{i=1}^{m}(Z_{1}w_{2i})\prod_{t=2}^{T}\exp(\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=Z_{1}\sum_{i=1}^{m}(w_{2i} \cdot \exp(\alpha_{2}y_{i}G_{2}(x_{i}))\prod_{t=3}^{T}\exp(\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=Z_{1}Z_{2}\sum_{i=1}^{m}(w_{3i} \prod_{t=3}^{T}\exp(\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
= \cdots
```
```math
= Z_{1}Z_{2} \cdots Z_{T-1}\sum_{i=1}^{m}w_{Ti} \cdot \exp(\alpha_{T}y_{i}G_{T}(x_{i}))
```
```math
= \prod_{t=1}^{T}Z_{t}
```
即
```math
\varepsilon \leqslant \frac{1}{m}\sum_{i=1}^{m}\exp(-y_{i}f(x_{i})) = \prod_{t=1}^{T}Z_{t}
```
这说明，我们可以在每一轮选取适当的$$G_{t}(x)$$，使得$$Z_{t}$$最小，从而使误差下降最快。进一步放大误差上界。
```math
Z_{t} = \sum_{i=1}^{m}w_{ti} \cdot \exp(-\alpha_{t}y_{i}G_{t}(x_{i}))
```
```math
=\sum_{y_{i} \neq G_{t}(x_{i})}w_{ti} \cdot \exp(a_{t}) + \sum_{y_{i} = G_{t}(x_{i})}w_{ti} \cdot \exp(-a_{t})
```
```math
=e_{t} \cdot \exp(a_{t}) + (1 - e_{t}) \cdot \exp(-a_{t})
```
```math
=2\sqrt{e_{t}(1-e_{t})} = \sqrt{1 - 4\gamma_{t}^{2}}
```
其中根据a_{t}的定义可得：$$\exp(a_{t}) = \sqrt{\frac{1-e_{t}}{e_{t}}}$$，$$\gamma_{t} = 2e_{t} -1$$。

根据$$e^{x}$$和$$\sqrt{1-x}$$的泰勒展开式可得：
```math
\prod_{t=1}^{T}\sqrt{1 - 4\gamma_{t}^{2}} \leqslant \exp(-2\sum_{t=1}^{T}\gamma_{t}^{2})
```
如果存在$$\gamma \gt 0$$，对于所有的$$t$$有$$\gamma_{t} \gt \gamma$$，则有
```math
\varepsilon \leqslant \exp(-2T\gamma_{t}^{2})
```
这表明，AdaBoost算法的误差是以指数的速度下降的。
