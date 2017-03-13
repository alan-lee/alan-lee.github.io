# softmax回归

softmax回归是logistic回归在多分类问题上的推广。softmax回归也是广义线性模型中的一种,其输出变量服从多项分布

考虑如下问题：
有一组数据集：

```math
\{(x^{(1)}, y^{(1)}), ((x^{(2)}, y^{(2)}), (x^{(3)}, y^{(3)}), ...(x^{(m)}, y^{(m)})\}
```
其中$$y^{(i)}\in\{1, 2, 3, \cdots k\}$$，和Logistic回归相比，其样本结果不在只是0，1两类，而是有k个分类，这就是多分类问题，显然其样本结果服从多项分布。

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
在Logistic回归中，我们假设$$\log\frac{\phi_{1}}{\phi_{0}} = \theta^{T}x$$，并且在对比广义线性模型后，发现其实自然参数$$\eta = \log\frac{\phi_{1}}{\phi_{0}}$$。既然softmax回归是Logistic回归的推广，那softmax应该也有类似的假设和结论。

假设：
```math
\log\frac{\phi_{i}}{\phi_{k}} = \theta^{T}_{i}x, \space i = 1, 2, \cdots, m - 1
```
那么
```math
\phi_{i} = \phi_{k}\exp(\theta^{T}_{i}x)
```
也就是说，以$$\phi_{k}$$作为参照，假设其他分类的概率和$$\phi_{k}$$的比率的对数和样本特征是线性的关系。

又因为
```math
\sum_{i=1}^{k}\phi_{i} = 1
\Rightarrow
```
```math
\phi_{k}(\sum_{i=1}^{k-1}\exp(\theta^{T}_{i}x) + 1) = 1
```

注意到$$\log\frac{\phi_{k}}{\phi_{k}} = 0$$，设有$$\theta^{T}_{k}x = 0$$，即$$\exp(\theta^{T}_{k}x) = 1$$，所以上式可以写成：
```math
\phi_{k}(\sum_{i=1}^{k}\exp(\theta^{T}_{i}x)) = 1
\Rightarrow
```
```math
\phi_{k} = \frac{1}{\sum_{i=1}^{k}\exp(\theta^{T}_{i}x)}
```
```math
\phi_{i} = \frac{\exp(\theta^{T}_{i}x)}{\sum_{i=1}^{k}\exp(\theta^{T}_{i}x)}
```

接下来利用最大似然估计函数来求解参数$$\theta_{i}$$的最大似然估计值：
```math
L(\theta) = \prod_{i=1}^{m}P(y^{(i)}| x^{(i)}, \theta)
```
```math
=\prod_{i=1}^{m}\prod_{j=1}^{k}\phi_{j}^{I(y^{(i)}=j)}
```
```math
=\prod_{i=1}^{m}\prod_{j=1}^{k}(\frac{\exp(\theta_{j}^{T}x^{(i)})}{\sum_{l=1}^{k}\exp(\theta_{l}^{T}x^{(i)})})^{I(y^{(i)}=j)}
```
取对数

```math
l(\theta) = \sum_{i=1}^{m}\sum_{j=1}^{k}I(y^{(i)}=j)(\theta_{j}^{T}x^{(i)} - \log({\sum_{j=1}^{k}\exp(\theta_{j}^{T}x^{(i)})}))
```

对$$l(\theta)$$求梯度有：
```math
\bigtriangledown_{\theta_{j}}l(\theta) = \sum_{i=1}^{m}(x^{(i)}(I(y^{(i)}=j) - P(y^{(i)} = j|x^{(i)},\theta)))
```

其中$$\triangledown_{\theta_{j}} l(\theta)$$本身也是一个向量，它的第$$l$$个元素$$\frac{\partial l(\theta)}{\partial \theta_{jl}}$$是$$l(\theta)$$对$$ \theta_{j} $$的第$$l$$个分量的偏导数。

有了以上的计算公式，我们就可以用梯度上升或是拟牛顿法来求解这一问题。

再看假设函数，我们把softmax的假设函数写成向量的形式，

```math
h_{\theta}(x)
=\left[\begin{matrix}
\phi_{1}
\\ \phi_{2}
\\ ...
\\ \phi_{k-1}
\\ \phi_{k}
\end{matrix}\right]
```
```math
=\left[\begin{matrix}
\frac{\exp(\theta_{1}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \frac{\exp(\theta_{2}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \vdots
\\ \frac{\exp(\theta_{k-1}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \frac{1}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\end{matrix}\right]
```

相对的样本的结果也需要写成向量的形式，最终，我们选择向量中最大的元素，其下标作为最后的分类。

# softmax回归与广义线性模型
文章开始时，我们提到了softmax回归是广义线性模型的一种。下面我们看看如何从广义线性模型来推导出我们上面的模型。

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
\\ \vdots
\\ \log\frac{\phi_{k-1}}{\phi_{k}}
\\ 0
\end{matrix}\right]
```

```math
T(y) =
\left[\begin{matrix}
I(y=1)
\\ I(y=2)
\\ \vdots
\\ I(y=k)
\end{matrix}\right]
```

由
```math
\eta =
\left[\begin{matrix}
\log\frac{\phi_{1}}{\phi_{k}}
\\ \log\frac{\phi_{2}}{\phi_{k}}
\\ \vdots
\\ \log\frac{\phi_{k-1}}{\phi_{k}}
\\ \log\frac{\phi_{k}}{\phi_{k}}
\end{matrix}\right]
\Rightarrow
```
```math
\eta_{i} =  \log\frac{\phi_{i}}{\phi_{k}}
```

由广义线性模型的自然参数$$\eta$$的假设知
```math
\eta =
\left[\begin{matrix}
\theta_{1}^{T}x
\\ \theta_{2}^{T}x
\\ \vdots
\\ \theta_{k}^{T}x
\end{matrix}\right]
```
则有
```math
\eta_{i} = \theta_{i}^{T}x
\Rightarrow
\log\frac{\phi_{i}}{\phi_{k}} = \theta_{i}^{T}x
```
这和我们模型是一致的，再看假设函数
```math
h_{\theta}(x)=E[T(y)|x;\theta]

=\left[\begin{matrix}
\phi_{1}
\\ \phi_{2}
\\ \vdots
\\ \phi_{k}
\end{matrix}\right]
```
```math
=\left[\begin{matrix}
\frac{\exp(\theta_{1}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \frac{\exp(\theta_{2}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \vdots
\\ \frac{\exp(\theta_{k-1}^{T}x)}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\\ \frac{1}{\sum_{j=1}^{k}\exp(\theta_{j}^{T}x)}
\end{matrix}\right]
```

# softmax回归和Logistic回归的关系
当k=2时，softmax回归就退化成了logistic回归

```math
h_{\theta}(x)
=\left[\begin{matrix}
\frac{\exp(\theta_{(1)}^{T}X)}{\exp(\theta_{(1)}^{T}X) + 1}
\end{matrix}\right]
=\frac{1}{1+\exp^{-{\theta^{T}X}}}
```

这就是logistic回归。
