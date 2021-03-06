# 朴素贝叶斯
朴素贝叶斯分类算法是一种很简单的分类算法，其基本原理就是根据贝叶斯条件概率计算出待分类项在各个分类中的概率，然后把概率最大的分类作为其类别，用数学语言来描述就是：

有待分类项$$x = \{x_{1}, x_{2}, x_{3}, \cdots, x_{n}\}$$和类别集合$$K = \{1, 2, 3, \cdots, k\}$$，$$x$$的类别$$y \in K$$，问题是要确定$$y$$的值。

解决问题的步骤是：
1.根据贝叶斯公式分别计算$$ p(y=1|x), \space p(y=2|x), \space p(y=3|x), \space p(y=k|x) $$
2.取$$p(y|x)$$的最大值时的y作为其分类: $$ y = \mathop{\arg\max} \limits_{y} p(y|x) $$

在第一步中，需要计算$$p(y=i|x)$$的条件概率，在计算时，需要用到贝叶斯公式：

```math
P(B|A) = \frac {P(AB)}{P(A)} = \frac{P(A|B) \cdot P(B)}{P(A)}
```

其中$$P(A|B)$$叫做先验概率，是指在在加入新的数据之前，初始化的概率或是通过经验得出结论为真的概率，是基于现有数据集的数据计算得出。$$P(B|A)$$叫做后验概率，是指加入新的数据后，计算得出的概率。

例如：

```math
p(y=i|x) = \frac {p(x, y=i)}{p(x)} = \frac {p(x | y=i)p(y = i)}{p(x)}
```
注意到$$x$$并一个独立的分布，而是$$n$$个事件的联合分布，因此有
```math
p(x) = p(x_{1})p(x_{2},x_{3},\cdots,x_{n}|x_{1}) = p(x_{1})p(x_{2})p(x_{3},x_{4},\cdots,x_{n}|x_{1},x_{2}) = \cdots
```
```math
= p(x_{1})p(x_{2})\cdots p(x_{n-1}) p(x_{n}|x_{1},x_{2},\cdots,x_{n-1})
```

这使得$$p(x)$$的计算相当负责，为了简化计算，在朴素贝叶斯算法中，有一个很重要的假设，就是$$x_{1}, x_{2}, \cdots, x_{n}$$这$$n$$个事件是独立事件，因此$$p(x)=p(x_{1})p(x_{2})\cdots p(x_{n})$$，这种计算就大大简化了。这也是朴素贝叶斯算法“朴素”的原因。

# 多项分布

多项分布在介绍softmax回归算法时已经详细介绍过了，对于

```math
x \sim Multinomial(\phi, k)
```
其概率为
```math
P(x) = \prod_{i=1}^{k}\phi_{i}^{I(x=i)}
```

# 垃圾邮件分类

垃圾分类时朴素贝叶斯算法最典型的应用，假定现在有$$m$$封邮件，其中是否是垃圾邮件已经标注出来，用$$1$$来表示邮件是垃圾邮件，用$$0$$表示是正常邮件。

首先，我们将这$$m$$封邮件做一些预处理
1.对每一份邮件做分词处理，然后把所有的词放在一起按照某一顺序(例如字母的升序)组成一个字典$$D$$，$$|D|$$表示字典中词的个数，即字典的长度，令$$l=|D|$$。
2.将字段中的每一个词建立索引，其值是该词在字典中出现的顺序，索引从1开始计数。
3.对于每一封邮件，我们其中的每个词替换成其在字典中索引，对于第$$i$$封邮件，可以得到一个长度为$$n_{i}$$向量$$x^{(i)} = [x^{(i)}_{1}, x^{(i)}_{2}, \cdots,x^{(i)}_{n_{i}}]$$，该邮件是否是垃圾邮件记为$$y^{(i)}$$，则$$y^{(i)} \in \{0, 1\}$$

这样，就得到了一组长度为$$m$$的数据集
```math
\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)}) \}
```

其中

```math
x^{(i)}_{j} \sim Multinomial(\phi_{x}, l)
```
```math
y^{(i)} \sim Bernoulli(\phi_{y})
```
任意一组数据$$(x^{(i)}, y^{(i)})$$都是独立分布。

由经验可知，同一词在正负样本中出现的概率是不同的，因此，我们有下面的定义：
```math
p(x^{(i)}_{j}|y=1) = \phi_{k|y=1}
```
```math
p(x^{(i)}_{j}|y=0) = \phi_{k|y=0}
```

模型已经有了，记下来就是策略和算法了，依然是利用最大似然估计函数来求解各个参数。

```math
L(\phi_{k|y=1}, \phi_{k|y=0}, \phi_{y}) = \prod_{i=1}^{m}p(x^{(i)}, y^{(i)}) = \prod_{i=1}^{m}p(x^{(i)} | y^{(i)}) \cdot p(y^{(i)})
```
```math
=\prod_{i=1}^{m}(\prod_{j=1}^{n_{i}}p(x^{(i)}_{j} | y^{(i)})) \cdot p(y^{(i)})
```

取对数
```math
l(\phi_{k|y=1}, \phi_{k|y=0}, \phi_{y}) = \sum_{i=1}^{m}\sum_{j=1}^{n_{i}}\log p(x^{(i)}_{j} | y^{(i)}) + \sum_{i=1}^{m}\log p(y^{(i)})
```
```math
= \sum_{i=1}^{m}\sum_{j=1}^{n_{i}} \sum_{h=1}^{l} {I(x^{(i)}_{j}=h)}\log \phi_{x} + \sum_{i=1}^{m}(y^{(i)}\log \phi_{y} + (1 - y^{(i)})log(1 - \phi_{y}))  
```
```math
= \sum_{i=1}^{m}\sum_{j=1}^{n_{i}}\sum_{h=1}^{l} I(y^{(i)} = 1) I(x^{(i)}_{j}=h) \log \phi_{h|y=1} + \sum_{i=1}^{m}\sum_{j=1}^{n_{i}}\sum_{h=1}^{l} I(y^{(i)} = 0) I(x^{(i)}_{j}=h) \log \phi_{h|y=0}  
```
```math
+ \sum_{i=1}^{m}(y^{(i)}\log \phi_{y} + (1 - y^{(i)})log(1 - \phi_{y}))
```

注意到上面等式的右边是由三部分相加组成，其中第一部分只和$$\phi_{k|y=1}$$相关，第二部分只和$$\phi_{k|y=0}$$相关，第三部分只和$$\phi_{y}$$相关，这样对这些参数求偏导数的时候就会简单一些。

```math
\frac{\partial{l(\phi_{k|y=1}, \phi_{k|y=0}, \phi_{y})}}{\partial{\phi_{k|y=1}}} = \sum_{i=1}^{m}I(y^{(i)} = 1) \sum_{j=1}^{n_{i}}(\frac {I(x^{(i)}_{j}=k)}{\phi_{k|y=1}} - \frac {I(x^{(i)}_{j}=l)}{\phi_{l|y=1}})
```

因为$$\phi_{l|y=1} = 1 -  \sum_{h=1}^{i-1} \phi_{h|y=1}$$，所以在计算偏导数时必须还要计算$$\phi_{l|y=1}$$对$$\phi_{1|y=1}, \phi_{2|y=1}, \cdots, \phi_{l-1|y=1}$$的导数。 要使最大似然估计函数取得最大值，则令上面的偏导数为0，可得：
```math
\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}\frac {I(y^{(i)} = 1)I(x^{(i)}_{j}=k)}{\phi_{k|y=1}} = \sum_{i=1}^{m} \sum_{j=1}^{n_{i}}\frac {I(y^{(i)} = 1)I(x^{(i)}_{j}=l)}{\phi_{l|y=1}}
```
即：
```math
\phi_{k|y=1} = \frac{\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 1)I(x^{(i)}_{j}=k)}{\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 1)I(x^{(i)}_{j}=l)}\cdot \phi_{l|y=1}
```

根据$$\sum_{h=1}^{l}\phi_{h|y=1} = 1$$可得：
```math
(\frac{\sum_{i=1}^{m} \sum_{j=1}^{n_{i}} \sum_{h=1}^{l-1}I(y^{(i)} = 1)I(x^{(i)}_{j}=h)}{\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 1)I(x^{(i)}_{j}=l)} + 1 )\cdot \phi_{l|y=1} = 1
```

```math
\phi_{l|y=1} = \frac {\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 1)I(x^{(i)}_{j}=l)}{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}I(y^{(i)} = 1)}
```
代入到上面的等式中得：
```math
\phi_{k|y=1} = \frac {\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 1)I(x^{(i)}_{j}=k)}{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}I(y^{(i)} = 1)}
```

同样可得：
```math
\phi_{k|y=0} = \frac {\sum_{i=1}^{m} \sum_{j=1}^{n_{i}}I(y^{(i)} = 0)I(x^{(i)}_{j}=k)}{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}I(y^{(i)} = 0)}
```

下面再来求$$\phi_{y}$$
```math
\frac{\partial{l(\phi_{k|y=1}, \phi_{k|y=0}, \phi_{y})}}{\partial{\phi_{y}}} = \sum_{i=1}^{m}(\frac{y^{(i)}}{\phi_{y}} - \frac{1 - y^{(i)}}{1 - \phi_{y}}) = \sum_{i=1}^{m}(\frac{I(y^{(i)} = 1)}{\phi_{y}} - \frac{I(y^{(i)} = 0)}{1 - \phi_{y}})
```

令上面的偏导数为0可得：
```math
\phi_{y} = \frac{\sum_{i=1}^{m}I(y^{(i)} = 1)}{m}
```

这和直观上得出的结果是一直的。

对于新样本$$x = (x_{1}, x_{2}, \cdots, x_{n})$$来说，

分别计算出$$p(y=1|x), p(y=0|x)$$后，再比较大小:
```math
p(x|y=1) = \prod_{i=1}^{n}p(x_{i}|y=1) = \prod_{i=1}^{n} \prod_{k=1}^{l}\phi_{k|y=1}^{I(x_{i}=k)}
```
```math
p(x|y=0) = \prod_{i=1}^{n}p(x_{i}|y=1) = \prod_{i=1}^{n} \prod_{k=1}^{l}\phi_{k|y=0}^{I(x_{i}=k)}
```
```math
p(y=1) = \phi_{y}
```
```math
p(y=0)=1 - \phi_{y}
```

由全概率公式得：
```math
p(x) = p(x|y=1)p(y=1) + p(x|y=0)p(y=0)
```

最后由贝叶斯公式计算出：
```math
p(y=1|x)=\frac{p(x|y=1)p(y=1)}{p(x)}
```
```math
p(y=0|x)=\frac{p(x|y=0)p(y=0)}{p(x)}
```
