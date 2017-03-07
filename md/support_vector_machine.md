
# 函数间隔和几何间隔

## 超平面
一般的线性分类算法，我们都可以画了一个分界线或是分界面来把正负样本分开，SVM算法也是一样，在本节和下一节中，我们都假定样本是线性可分的，也就是可以用一个线(2维)或面(3维或是更高维)来来把正负样本分开，这个面就叫做超平面(hyperplane)。超平面的定义如下：
对于数据集中的任意一组样本，平面$$w^{T}x + b$$，有
```math
w^{T}x + b \gt 0 \space \Leftrightarrow y^{(i)} = 1
```
```math
w^{T}x + b \lt 0 \space \Leftrightarrow y^{(i)} = -1
```
注意在之前介绍的二分类算法中，我们用$$1$$来表示正样本，用$$0$$来表示负样本。但是在SVM算法中，我们用$$1$$来表示正样本，用$$-1$$来表示负样本。

即SVM算法的假设函数
```math
h_{w.b} = g(w^{T}x + b)
```
```math
g(x) = \left\{\begin{matrix}
0\qquad x \gt 0
\\1\qquad x \lt 0
\end{matrix}\right.
```

## 函数间隔
在下图中，有两类数据，图中画了3条直线(超平面)可以把这两类数据分开，那哪条直线才是最优的呢？

![image](http://oirw7ycq0.bkt.clouddn.com/svm.png)

通过观察可以发现，黄线是最优的，相比绿线和红线，它离正负样本的距离都相对远一些，这样的分隔线对新样本来说，预测的结果会更准确。那如何将观察的结果用数学来表达呢？首先，要引入函数间隔的概念。

对于某一样本$$(x^{(i)}, y^{(i)})$$，定义其函数间隔为
```math
\hat{\gamma}^{(i)} = y^{(i)}(w^{T}x + b)
```
对整个样本来说，
```math
    \hat{\gamma} = \mathop{\min} \limits_{i=1 \cdots m} \hat{\gamma}^{(i)}
```
若$$(x^{(i)}, y^{(i)})$$在超平面上，则有$$\hat{\gamma}_{(i)}=0$$，当$$(x^{(i)}, y^{(i)})$$距离超平面越远，则$$|w^{T}x + b|$$的值也就越大，因此可以用$$|w^{T}x + b|$$来衡量某点距离超平面的距离。
根据定义，若$$w^{T}x + b < 0$$，则$$y^{(i)} = -1$$，若$$w^{T}x + b > 0$$，则$$y^{(i)} = -1$$，因此有$$\hat{\gamma}^{(i)} = y^{(i)}(w^{T}x + b)$$。

直观上，我们可以用函数间隔来表示某点距离超平面的距离，而且距离越大，我们越能确定其分类的正确性。但是函数间隔存一个问题，当我们把$$w$$和$$b$$做等比放大时，$$\hat{\gamma}$$也会变大，但这不能说明该样本分类的正确性提高了。为了解决这个问题，我们引入几何间隔的概念。

## 几何间隔
几何间隔的定义如下：
```math
\gamma = \frac{\hat{\gamma}}{\|w\|}
```
其中$$\|w\|$$叫做向量$$w$$的模，L2范式，欧式长度，$$\|w\| = \sqrt{\sum_{i=1}^{n}w_{j}^{2}}$$
从公式上看，几何间隔等于函数间隔除上向量$$w$$的模，相当于对函数间隔做了正规化。此外，几何间隔在几何中还能表示$$n$$维空间中， 点$$x^{(i)}$$到超平面的垂直距离。

在下图上，$$A$$点表示$$x^{(i)}$$，直线L表示超平面，$$B$$点是点$$A$$在$$l$$上的投影，设$$\gamma^{(i)} = \|AB\|$$，下面求解$$\gamma^{(i)}$$的值。

![image](http://oirw7ycq0.bkt.clouddn.com/svm_margin.png)

令超平面为$$f(x) = w^{T}x + b$$，先求$$f(x)$$对$$x$$的梯度：
```math
\bigtriangledown_{x}f(x) = w
```
根据法向量的定义知：$$w$$即为超平面的法向量，$$\frac{w}{\|w\|}$$则为超平面的单位法向量。

又有直线$$AB$$和$$l$$也垂直，所以有$$\mathop{AB} \limits^{\rightharpoonup} = \gamma^{(i)} \cdot \frac{w}{\|w\|} $$

由向量的加减法知：$$\mathop{OB} \limits^{\rightharpoonup} = \mathop{AB} \limits^{\rightharpoonup} - \mathop{OA} \limits^{\rightharpoonup}$$

所以$$\mathop{OB} \limits^{\rightharpoonup} = x - \gamma^{(i)} \cdot \frac{w}{\|w\|}$$

由因为$$B$$点在直线$$l$$上，所以有：
```math
w^{T} \cdot (x^{(i)} - \gamma^{(i)} \cdot \frac{w}{\|w\|}) + b = 0
```
```math
w^{T}x^{(i)} - \gamma^{(i)} \cdot\frac{w^{T}w}{\|w\|} + b = 0
```
```math
\gamma^{(i)} \cdot \frac{\|w\|^{2}}{\|w\|} = w^{T}x^{(i)} + b
```
```math
\gamma^{(i)} \cdot \|w\| = \hat{\gamma}^{(i)}
```
```math
\gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\|w\|}
```

同样，我们定义整个数据集的几何间隔为
```math
    \gamma = \mathop{\min} \limits_{i=1 \cdots m} \gamma^{(i)}
```

# 最大化间隔分类算法
根据上面的分析，我们已经给问题建立好模型了，所谓最大化间隔分类算法，就是要找到一组参数，使得数据集的几何间隔$$\gamma$$最大。

## 最大化间隔的原始问题
我们要解决的原始问题是：
```math
f(w,b) = \max \limits_{w,b} \gamma = \max \limits_{w,b} \frac{\hat{\gamma}}{\|w\|}
```
```math
st. \quad y^{(i)}(w^{T}x^{(i)} + b) \geqslant \hat{\gamma}, \space i = 1, 2, \cdots, m
```

$$f(w,b)$$并不是一个凸函数，因此这个问题并不好解，在前面介绍函数间隔时，我们说到，我们可以对$$w,b$$进行等比缩放，而不会影响最终样本分类的结果。所以，这里我们可以假设通过缩放得到$$\hat{\gamma} = 1$$，那么上面的问题可以转化为：
```math
f(w,b) = \min \limits_{w,b} \frac{1}{2}\|w\|^{2}
```
```math
s.t. \quad y^{(i)}(w^{T}x^{(i)} + b) \geqslant 1, \space i = 1, 2, \cdots, m
```
注意求$$\frac{1}{\|w\|}$$的最大值即求$$\|w\|^{2}$$的最小值。上面加上$$\frac{1}{2}$$是为了后面的计算方便。

新的问题是一个二次规化的问题，利用拉格朗日对偶性将该问题转化成对偶问题，可以通过求解对偶问题来得到原始问题的解。

## 拉格朗日对偶性
在求解一些带约束条件的最优化问题时，常常利用拉格朗日对偶性将该问题转化成对偶问题，可以通过求解对偶问题来得到原始问题的解。

### 原始问题

假设$$f(x), c_{i}(x), h_{j}(x)$$都是定义在$$R^{n}$$上连续可微的函数，考虑一下最优化问题：
```math
\min \limits_{x \in R^{n}} f(x)
```
```math
s.t. \quad c_{i}(x) \leqslant 0, \space i=1, 2, \cdots, k
```
```math
\space \qquad h_{j}(x) = 0, \space j=1, 2, \cdots, l
```
该约束最优化问题称之为原始问题。

首先，引进广义拉格朗日函数
```math
L(x, \alpha, \beta) = f(x) + \sum_{i=1}^{k}\alpha_{i}c_{i}(x) + \sum_{j=1}^{l}\beta_{j}h_{j}(x)
```

这里$$x = (x_{1}, x_{2}, \cdots, x_{n})^{T} \in R^{n}$$, $$\alpha_{i}, \beta_{j}$$是拉格朗日乘数，并且有$$\alpha_{i} \geqslant 0$$，考虑$$x$$的函数：
```math
\theta_{P}(x) = \max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
```
若原始问题的约束条件成立，则$$\alpha_{i}c_{i}(x) \leqslant 0$$, $$\beta_{j}h_{j}(x) = 0$$
要使$$L(x, \alpha, \beta)$$取得最大值，只有令$$\alpha_{i}c_{i}(x) = 0$$，这样有$$\theta_{P}(x) = f(x)$$

若原始问题的约束条件不成立，则存在$$\alpha_{i} > 0$$或$$h_{j}(x) \neq 0$$，那么只要令$$\alpha_{i} \rightarrow \infty$$或$$h_{j}(x) \rightarrow \infty$$则$$\theta_{P}(x) \rightarrow \infty$$，所以有：
```math
\theta_{P}(x) = \left\{\begin{matrix}
f(x) & x \text{ satisfies primal constraints }
\\1 & \text{otherwise}
\end{matrix}\right.
```
再考虑最小化问题，
```math
\min \theta_{P}(x) = \min \limits_{x} \max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
```
在满足原始约束条件下，它和原始问题是等价的，即有相同的解。问题$$\min \limits_{x} \max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} L(x, \alpha, \beta)$$称为广义拉格朗日函数的极小极大问题，这样就把原始问题转化为广义拉格朗日函数的极小极大问题。

### 对偶问题
定义
```math
\theta_{D}(\alpha, \beta) = \min \limits_{x}L(x, \alpha, \beta)
```

再考虑$$\theta_{D}(\alpha, \beta)$$的最大化问题
```math
\max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} \theta_{D}(\alpha, \beta) = \max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} \min \limits_{x}L(x, \alpha, \beta)
```
这一问题称为广义拉格朗日函数的极大极小问题。
定义
```math
\max \limits_{\alpha, \beta} \min \limits_{x}L(x, \alpha, \beta)
```
```math
s.t. \quad \alpha_{i} \geqslant 0, \space i=1, 2, \cdots, k
```

为原始问题的对偶问题。
### KKT条件
下面讨论原始问题和对偶问题的关系。

令
```math
p^{*} = \min \theta_{P}(x)
```
```math
d^{*} = \max \theta_{D}(\alpha, \beta)
```
那么有：
```math
d^{*} \leqslant p^{*}
```
这个不等于很好推导，对于任意的$$\alpha, \beta, x$$都有
```math
\theta_{D}(x) = \min \limits_{x}L(x, \alpha, \beta) \leqslant L(x, \alpha, \beta) \leqslant \max \limits_{\alpha, \beta, \alpha_{i} \geqslant 0} L(x, \alpha, \beta) = \theta_{P}(x)
```

所以
```math
\max \theta_{D}(\alpha, \beta) \leqslant \min \theta_{P}(x)
```

若存在$$\alpha^{*}, \beta^{*}, x^{*} $$, 使得
```math
L(\alpha^{*}, \beta^{*}, x^{*}) = d^{*} = p^{*}
```
那么$$\alpha^{*}, \beta^{*}$$和$$x^{*}$$分别是对偶问题和原始问题的最优解。

这很好理解，因为$$d^{*} \leqslant p^{*}$$，所以$$p^{*}$$的下限就是$$d^{*}$$，若有$$x^{*}$$使得$$\theta_{P}(x) = d^{*}$$，那$$\theta_{P}(x^{*})$$一定是$$\theta_{P}(x)$$的极小值，即是原始问题的最优解。同理可知$$\alpha^{*}, \beta^{*}$$是对偶问题的最优解。

当$$d^{*} = p^{*}$$时，可以用解对偶问题来代替解原始问题。只有当$$\alpha^{*}, \beta^{*}, x^{*} $$满足一定的条件时，才有$$d^{*} = p^{*}$$。

对于原始问题和对偶问题，假设函数$$f(x)$$和$$c_{i}(x)$$是凸函数，$$h_{j}(x)$$是仿射函数，并且不等式约束$$c_{i}(x)$$是严格可行的，即存在$$x$$，对所有的$$i$$有$$c_{i}(x) \lt 0$$，则$$x^{*}$$和$$\alpha^{*}, \beta^{*}$$分别是原始问题和对偶问题的解的充分必要条件是满足下面的Karush-Kuhn-Tucker(KKT)条件：
```math
\bigtriangledown_{x}L(\alpha^{*}, \beta^{*}, x^{*}) = 0
```
```math
\bigtriangledown_{\alpha}L(\alpha^{*}, \beta^{*}, x^{*}) = 0
```
```math
\bigtriangledown_{\beta}L(\alpha^{*}, \beta^{*}, x^{*}) = 0
```
```math
\alpha^{*}_{i}c_{i}(x^{*}) = 0, \quad i = 1, 2, \cdots, k
```
```math
c_{i}(x^{*}) \leqslant 0,  \quad i = 1, 2, \cdots, k
```
```math
\alpha^{*}_{i} \geqslant 0,  \quad i = 1, 2, \cdots, k
```
```math
h_{j}(x^{*}) = 0, \quad j = 1, 2, \cdots, l
```

上面第4个等式称为KKT对偶互补条件，由此条件可知：若$$\alpha^{*}_{i} \gt 0$$, 则$$c_{i}(x^{*}) = 0$$

### SVM的对偶性
之前我们已经给出最大间隔的原始问题：
```math
\min \limits_{w,b} \frac{1}{2}\|w\|^{2}
```
```math
s.t. \quad y^{(i)}(w^{T}x^{(i)} + b) \geqslant 1, \space i = 1, 2, \cdots, m
```

为了看上去更像KKT条件，我们把不等于约束写成：
```math
c_{i}(w, b) = 1 - y^{(i)}(w^{T}x^{(i)} + b) \leqslant 0, \space i = 1, 2, \cdots, m
```

构建拉格朗日函数
```math
L(\alpha, w, b) = \frac{1}{2}\|w\|^{2} - \sum_{i=1}^{m}\alpha_{i}( y^{(i)}(w^{T}x^{(i)} + b) - 1)
```

若满足KKT的对偶互补条件，并且某$$\alpha_{i} \neq 0$$，则有$$y^{(i)}(w^{T}x^{(i)} + b) = 1$$

上面假设几何间隔$$\gamma = 1$$，根据几何间隔的定义，点$$x^{(i)}$$就是离超平面最近的点，这样的点叫做支持向量。

首先求$$L(\alpha, w, b)$$对参数$$w,b$$的极小值：
```math
\bigtriangledown_{w}L(\alpha, w, b) = w - \sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)} = 0
```

可得
```math
w = \sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)}
```

```math
\bigtriangledown_{b}L(\alpha, w, b) = \sum_{i=1}^{m}\alpha_{i}y^{(i)} = 0
```

将上面的等式待入到拉格朗日函数：
```math
\min \limits_{w, b} L(\alpha, w, b) = \frac{1}{2}\|w\|^{2} - \sum_{i=1}^{m}\alpha_{i}( y^{(i)}(w^{T}x^{(i)} + b) - 1)
```
```math
= \frac{1}{2}w^{T}w - \sum_{i=1}^{m}\alpha_{i}y^{(i)}w^{T}x^{(i)} - b\sum_{i=1}^{m}\alpha_{i}y^{(i)} + \sum_{i=1}^{m}\alpha_{i}
```
```math
=\frac{1}{2}w^{T}(\sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)}) - w^{T}(\sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)}) + \sum_{i=1}^{m}\alpha_{i}
```
```math
=-\frac{1}{2}w^{T}(\sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)}) + \sum_{i=1}^{m}\alpha_{i}
```
```math
=-\frac{1}{2}(\sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)})^{T}(\sum_{i=1}^{m}\alpha_{i}y^{(i)}x^{(i)}) + \sum_{i=1}^{m}\alpha_{i}
```
```math
=\sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)}
```
上面的式子中已经没有$$w,b$$了，这样，在满足KKT条件时，可以把原始问题，转化为下面的对偶问题
```math
\max \limits_{\alpha} \sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)}
```
```math
s.t. \quad \alpha_{i} \geqslant 0, \space i=1, 2, \cdots, m
```
```math
\sum_{i=1}^{m}\alpha_{i}y^{(i)} = 0
```

这样，求出了$$\alpha$$，就可以计算出$$w$$，计算$$b$$的过程稍有些复杂，对于超平面$$w^{T}x + b$$来说，为了使分类的预测更加准确，正负样本中的支持变量的几何间隔一定使相同的，假设$$w$$的最优解是$$w^{*}$$, $$b$$的最优解是$$b^{*}$$，则有：
```math
\max \limits_{i:y^{(i)} = -1} (w^{*})^{T}x^{(i)} + b = \min \limits_{i:y^{(i)} = 1} (w^{*})^{T}x^{(i)} + b
```
```math
b = - \frac{\max \limits_{i:y^{(i)} = -1} (w^{*})^{T}x^{(i)} + \min \limits_{i:y^{(i)} = 1} (w^{*})^{T}x^{(i)}}{2}
```

在计算出$$w,b$$之后，最可以根据
```math
w^{T}x + b = \sum_{i=1}^{m}\alpha_{i}y^{(i)}(x^{(i)})^{T}x + b
```
来判断新的样本的分类了，其中，只有支持向量的$$\alpha_{i} \neq 0$$，所以大量的非支持向量都不需要计算，所以$$w^{T}x + b$$的计算效率会很高。

上面已经给出了SVM的对偶问题，并且分析了在求得对偶问题的解后，如何来预测新样本的分类。求解对偶问题中的拉格朗日算数$$\alpha$$，最常用的算法就是SMO，后文中会的详细介绍这一算法。下面先对如何处理线性不可分的数据集做一些讨论。

# 线性不可分和软间隔
以上的最大间隔分类算法对于线性不可分的样本集是不适用的，因为上述的约束不等于并不是对于数据集中的所有点都能成立。一般的情况是，样本集中的少数的特异点，去掉这些样本后，剩下的大部分样本组成的数据集是线性可分的。

为了解决这个问题，我们对每个样本$$(x^{(i)}, y^{(i)})$$引入一个松弛变量$$\xi_{i} \geqslant 0$$，这样，约束条件变成了
```math
y^{(i)}(w^{T}.x^{(i)} + b) \geqslant 1 - \xi_{i}, \quad i = 1, 2, \cdots, m
```
同样，为了使分类更为准确，我们希望$$\xi_{i}$$越小越好，所以在目标函数后加上一个L1范数正规化选项:
```math
\frac{1}{2} \|w\|^{2} + C\sum_{i=1}^{m}\xi_{i}
```
这里的$$C \gt 0$$称为惩罚参数，$$C$$的值大时，对特异点的惩罚加大，$$C$$的值小时，对特异点的惩罚变大，最小化新的目标函数，一方面是要最小化几何间隔，另一方面，可要使特异点尽可能的稍，参数C就是用来调整两者的权重的。

注意到当$$\xi_{i} = 0$$，问题就退化成线性可分数据集的情况了。这种处理线性不可分数据集的SVM算法称为软间隔最大化算法。

软间隔最大化算法的原始问题为：
```math
\min \limits_{w, b, \xi} \frac{1}{2} \|w\|^{2} + C\sum_{i=1}^{m}\xi_{i}
```
```math
s.t. \quad y^{(i)}(w^{T}x^{(i)} + b) \geqslant 1 - \xi_{i}, \quad i = 1, 2, \cdots, m
```
```math
\xi_{i} \geqslant 0, \quad i = 1, 2, \cdots, m
```

原始问题的拉格朗日函数
```math
L(w, b, \xi, \alpha, \beta) = \frac{1}{2} \|w\|^{2} + C\sum_{i=1}^{m}\xi_{i} - \sum_{i=1}^{m}\alpha_{i}( y^{(i)}(w^{T}x^{(i)} + b) - 1 + \xi_{i}) - \sum_{i=1}^{m}\beta_{i}\xi_{i}
```
其中$$\alpha_{i} \geqslant 0,  \beta_{i} \geqslant 0$$，首先求$$L(w, b, \xi, \alpha, \beta)$$对$$w,b,\xi$$的极小：

```math
\bigtriangledown_{w}L(w, b, \xi, \alpha, \beta) = w - \sum_{i=1}^{m} \alpha_{i}y^{(i)}x^{(i)} = 0
```
即
```math
w = \sum_{i=1}^{m} \alpha_{i}y^{(i)}x^{(i)}
```
```math
\bigtriangledown_{b}L(w, b, \xi, \alpha, \beta) = - \sum_{i=1}^{m} \alpha_{i}y^{(i)} = 0
```
```math
\bigtriangledown_{\xi}L(w, b, \xi, \alpha, \beta) = C - \alpha - \beta = 0
```

代入到$$L(w, b, \xi, \alpha, \beta)$$中得：
```math
\min \limits_{w, b, \xi} L(w, b, \xi, \alpha, \beta) = \sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)}
```

可得原始问题的对偶问题为：
```math
\max \limits_{\alpha, \beta} \sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)}
```
```math
s.t. \quad \sum_{i=1}^{m} \alpha_{i}y^{(i)} = 0, i = 1, 2, \cdots, m
```
```math
C - \alpha_{i} - \beta_{i} = 0, i = 1, 2, \cdots, m
```
```math
\alpha_{i} \geqslant 0, i = 1, 2, \cdots, m
```
```math
\beta_{i} \geqslant 0, i = 1, 2, \cdots, m
```

再消去约束条件中的$$\beta_{i}$$，得新的约束条件：
```math
s.t. \quad 0 \leqslant \alpha_{i} \leqslant C, i = 1, 2, \cdots, m
```
其KKT对偶互补条件为：
```math
\alpha_{i}(1 - \xi_{i} - (w^{T}x^{(i)} + b)) = 0, i = 1, 2, \cdots, m
```
```math
\beta_{i}\xi_{i} = 0, i = 1, 2, \cdots, m
```

和线性可分数据集的算法一样，在求得拉格朗日乘数$$\alpha$$之后，可以计算出$$w,b$$，再根据$$h(x) = g(w^{T}x + b)$$来预测新样本的分类了。
# Hinge Loss
在介绍其他的线性学习算法时，我们都提到了损失函数(Loss Function)，对于SVM来说，也有损失函数：
```math
J(w, b) = \sum_{i=1}^{m}l(w^{T}x^{(i)} + b) + \lambda \|w\|^{2}
```

其中，对于样本$$x^{(i)}$$和其预测结果$$y^{(i)} = \pm 1$$，

```math
l(t(x^{(i)})) = \left\{\begin{matrix}
0 &  t(x^{(i)}) \leqslant 0
\\1 - y^{(i)}t(x^{(i)}) & t(x^{(i)}) \gt 0
\end{matrix}\right.
```
$$l(t)$$称为hinge损失函数。

可以证明，$$\min \limits_{w,b} J(w,b)$$和 $$\min \limits_{w,b,\xi} \frac{1}{2} \|w\|^{2} + C\sum_{i=1}^{m}\xi_{i}$$是等价的。
# 核函数
## 向量内积
有向量$$x,y$$，定义其内积运算$$\left \langle x \cdot y \right \rangle = x^{T}y = y^{T}x$$，显然，向量的内积运算的结果是一个标量。

我们可以把上面的分析得到的对偶问题写成内积的形式:
```math
\max \limits_{\alpha} \sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)} \left \langle x^{(i)} \cdot x^{(j)} \right \rangle
```
在预测分类时，也可以利用内积：
```math
w^{T}x + b = \sum_{i=1}^{m}\alpha_{i}y^{(i)} \left \langle x^{(i)} \cdot x \right \rangle
```
## 核技巧
上面我们讨论的样本线性集不可分的问题主要是针对样本集中有一些特异点的情况，还有一种情况是样本集中的数据无法通过一个平面将其分开，但是可以通过一个曲面将其分开，也就是说可以用非线性模型来分类。

对于这类问题，我们可以通过一些函数变换，把非线性模型变换成线性模型。例如，已有数据集$${(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})}$$，其中$$x \in R^{2}$$，
已知该数据集中的两个类别可以被椭圆$$w_{1}x_{1}^{2} + w_{2}x_{2}^{2} + b =0$$分来。我们令$$z=\phi(x)=(x_{1}^{2}, x_{2}^{2})$$，把原空间的数据映射到新空间中，原空间中椭圆方程在新空间中就变成了$$w_{1}z_{1} + w_{2}z_{2} + b =0$$，这表示一条直线，这样数据集就变成线性可分的了。

核技巧就是用函数变换把原空间的数据映射到新的空间中，使得在原模型在新空间中变成一个线性问题，然后通过线性学习算法来求解。

大多数时候，我们并不能很简单的得到映射函数$$\phi(x)$$，或者是映射函数很复杂，会把原空间2维，3维的数据映射成$$n$$维或是无限维，这样在计算内积运算时，会变得非常复杂。
为了解决这一问题，我们可以并不显示的定义映射函数$$\phi(x)$$，而是用核函数来计算内积。核函数的定义如下：

对于输入空间$$X$$，设$$H$$为特征空间(希尔伯特空间)，如果存在一个从$$X \rightarrow H$$的映射$$\phi(x)$$，使得对所有的$$x, z \in X$$，函数$$K(x, z)$$都满足：
```math
K(x, z) = \left \langle \phi(x) \cdot \phi(z) \right \rangle
```
则称$$K(x, z)$$为核函数，$$\phi(x)$$为映射函数。

一般来说，我们在学习和预测的过程中，只会定义核函数$$K(x, z)$$，并不会显示的定义映射函数$$\phi(x)$$，这是因为直接计算$$K(x, z)$$比通过$$\phi(x)$$来计算$$K(x, z)$$要简单一些。

## 核函数的正定性
有样本集$$x^{(1)}, x^{(2)}, \cdots, x^{(m)} \in R^{n}$$, $$K$$是$$R^{n} \times R^{n} \rightarrow R$$的一个映射。
令矩阵
```math
K = \begin{bmatrix}
 K(x^{(1)}, x^{(1)}) & K(x^{(1)}, x^{(2)}) & \cdots & K(x^{(1)}, x^{(m)}) \\
 K(x^{(2)}, x^{(1)}) & K(x^{(2)}, x^{(2)}) & \cdots & K(x^{(2)}, x^{(m)}) \\
 \vdots & \vdots & \ddots &\vdots \\
 K(x^{(m)}, x^{(1)}) & K(x^{(m)}, x^{(2)}) & \cdots & K(x^{(m)}, x^{(m)}) \\
\end{bmatrix}
```
函数$$K$$是一个有效核函数的充要条件是矩阵$$K$$是一个对称半正定矩阵。
其中，矩阵$$K$$叫做核函数矩阵。

## 常用的核函数
1. 线性核函数
```math
K(x, z) = c\left \langle x \cdot z \right \rangle +d
```
2. 多项式核函数
```math
K(x, z) = (c\left \langle x \cdot z \right \rangle +d)^{p}
```
3. 高斯核函数
```math
K(x, z) = \exp(-\frac{\|x-z\|^{2}}{2\sigma^{2}})
```
高斯核函数是径向基核函数的一种，径向基核函数是指由特定点间的距离决定其值的函数。

4. Sigmoid核函数
```math
K(x, z) = \tanh(c\left \langle x \cdot z \right \rangle +d)
```
