# softmax回归

softmax回归是logistic回归在多分类问题上的推广。softmax回归也是广义线性模型中的一种,其输出变量服从多项分布

考虑如下问题：
有一组数据集：

```math
\{(x^{(1)}, y^{(1)}), ((x^{(2)}, y^{(2)}), (x^{(3)}, y^{(3)}), ...(x^{(m)}, y^{(m)})\}
```
其中$$y^{(i)}\in\{1, 2, 3, \cdots k\}$$，和logistic回归相比，其输出变量不在只是0，1两类，而是有k个分类，这就是多分类问题，显然其输出变量服从多项分布。

令$$P(y=i)=\phi_{i}$$ ,则有$$\phi_{k}=1-\sum^{k-1}_{i=1}\phi_{i}$$ 并指示函数用$$I(y=i)$$来指示$$y=i$$这个逻辑条件是否为真，若$$y=i$$为真，则$$I(y=i)$$的值是1，否则是0，即
```math
I(y=i) = \left\{\begin{matrix}
1,  \quad y =  i
\\ 0, \quad y \neq i
\end{matrix}\right.

I(y=k) = 1 - \sum_{i=1}^{k-1}I(y=i)
```
那么
```math
P(y) = \prod_{i=1}^{k}\phi_{i}^{I(y=i)}
```

softmax回归的模型已经介绍完了，下面来推到softmax回归的预测函数$$h_{\theta}(x)$$,
这里我们讲输出写成另外一种形式，令
```math
T(y) =
\left[\begin{matrix}
I(y=1)
\\ I(y=2)
\\ ...
\\ I(y=k)
\end{matrix}\right]
```
由广义线性模型知：

```math
 P(y) = \prod_{i=1}^{k}\phi_{i}^{I(y=i)}

 = \exp(\log\prod_{i=1}^{k}\phi_{i}^{I(y=i)})

 = \exp(\sum_{i=1}^{k}I(y=i)\log\phi_{i})
 ```
 ```math
 = \exp(\sum_{i=1}^{k-1}I(y=i)\log\phi_{i} + (1 - \sum_{i=i}^{k-1}I(y=i))\log\phi_{k})
 ```
 ```math
 = \exp(\sum_{i=1}^{k-1}I(y=i)\log\frac{\phi_{i}}{\phi_{k}} + \log\phi_{k})

 = \exp(\eta^{T} T(y) + \log\phi_{k})
```

其中
```math
\eta =
\left[\begin{matrix}
\log\frac{\phi_{1}}{\phi_{k}}
\\ \log\frac{\phi_{2}}{\phi_{k}}
\\ ...
\\ \log\frac{\phi_{k-1}}{\phi_{k}}
\end{matrix}\right]
```
向量$$T(y)$$取前$$k-1$$个元素，因为$$T(y)_{k}$$可以由前$$k-1$$个元素来表示。

由
```math
\eta =
\left[\begin{matrix}
\log\frac{\phi_{1}}{\phi_{k}}
\\ \log\frac{\phi_{2}}{\phi_{k}}
\\ ...
\\ \log\frac{\phi_{k-1}}{\phi_{k}}
\end{matrix}\right]
\Rightarrow
```
```math
\eta_{i} =  \log\frac{\phi_{i}}{\phi_{k}}
\Rightarrow

\phi_{i} = \phi_{k} \cdot \exp(\eta_{i})
```
又有
```math
\sum_{i=1}^{k}\phi_{i} = 1
\Rightarrow

\phi_{k}(\sum_{i=1}^{k-1}\exp(\eta_{i}) + 1) = 1
\Rightarrow
```
```math
\phi_{k} = \frac{1}{\sum_{i=1}^{k-1}\exp(\eta_{i}) + 1}
\Rightarrow

\phi_{i} = \frac{\exp(\eta_{i})}{\sum_{j=1}^{k-1}\exp(\eta_{j}) + 1}
```

由广义线性模型的假设知
```math
\eta =
\left[\begin{matrix}
\theta_{(1)}^{T}X
\\ \theta_{(2)}^{T}X
\\ ...
\\ \theta_{(k-1)}^{T}X
\end{matrix}\right]
```
则有
```math
\eta_{i} = \theta_{(i)}^{T}X
\Rightarrow

\phi_{i} = \frac{\exp(\theta_{(i)}^{T}X)}{\sum_{j=1}^{k-1}\exp(\theta_{(j)}^{T}X) + 1}
```
又有
```math
h_{\theta}(x)=E[T(y)|x;\theta]

=\left[\begin{matrix}
\phi_{1}
\\ \phi_{2}
\\ ...
\\ \phi_{k-1}
\end{matrix}\right]
=\left[\begin{matrix}
\frac{\exp(\theta_{(1)}^{T}X)}{\sum_{j=1}^{k-1}\exp(\theta_{(j)}^{T}X) + 1}
\\ \frac{\exp(\theta_{(2)}^{T}X)}{\sum_{j=1}^{k-1}\exp(\theta_{(j)}^{T}X) + 1}
\\ ...
\\ \frac{\exp(\theta_{(k-1)}^{T}X)}{\sum_{j=1}^{k-1}\exp(\theta_{(j)}^{T}X) + 1}
\end{matrix}\right]
```

接下来就是用一组$$\theta$$来拟合给定的数据集了，首先是最大似然估计函数：
```math
L(\theta) = \prod_{i=1}^{m}P(y^{(i)}| x^{(i)}, \theta)

=\prod_{i=1}^{m}\prod_{j=1}^{k}\phi_{j}^{I(y=j)}

=\prod_{i=1}^{m}\prod_{j=1}^{k}(\frac{\exp(\theta_{(j)}^{T}X)}{\sum_{l=1}^{k-1}\exp(\theta_{(l)}^{T}X) + 1})^{I(y=j)}
```
取对数

```math
l(\theta) = \sum_{i=1}^{m}\sum_{j=1}^{k}I(y=j)(\theta_{(j-1)}^{T}X - \log({\sum_{j=1}^{k-1}\exp(\theta_{(j)}^{T}X) + 1}))
```

对$$l(\theta)$$求梯度有：
```math
\bigtriangledown_{\theta_{j}}l(\theta) = \sum_{i=1}^{m}(x^{(i)}(I(y^{(i)}=j) - P(y^{(i)} = j|x^{(i)},\theta)))
```

其中$$\triangledown_{\theta_{j}} l(\theta)$$本身也是一个向量，它的第$$l$$个元素$$\frac{\partial l(\theta)}{\partial \theta_{jl}}$$是$$l(\theta)$$对$$ \theta_{j} $$的第$$l$$个分量的偏导数。

有了以上的计算公式，我们就可以用梯度上升获释牛顿法来求解这一问题。

# softmax回归和logistic回归的关系
当k=2时，softmax回归就退化成了logistic回归

```math
h_{\theta}(x)
=\left[\begin{matrix}
\frac{\exp(\theta_{(1)}^{T}X)}{\exp(\theta_{(1)}^{T}X) + 1}
\end{matrix}\right]
=\frac{1}{1+\exp^{-{\theta^{T}X}}}
```

这就是logistic回归。
