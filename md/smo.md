SMO算法的全称是Sequential Minimal Optimization，1998年由微软的John.C.Platt提出，是目前求解SVM问题最有效的算法。

SVM的原始问题是：
```math
\min \limits_{w, b, \xi} \frac{1}{2} \|w\|^{2} + C\sum_{i=1}^{m}\xi_{i}
```
```math
s.t. \quad y^{(i)}(w^{T}.x^{(i)} + b) \geqslant 1 - \xi_{i}, \quad i = 1, 2, \cdots, m
```
```math
\xi_{i} \geqslant 0, \quad i = 1, 2, \cdots, m
```
利用拉格朗日函数的对偶性，可以得到其对偶问题：
```math
\max \limits_{\alpha} \sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)}
```
```math
s.t. \quad \sum_{i=1}^{m} \alpha_{i}y^{(i)} = 0, i = 1, 2, \cdots, m
```
```math
0 \leqslant \alpha_{i} \leqslant C, i = 1, 2, \cdots, m
```
只要求解出上面问题上的拉格朗日乘数$$\alpha$$，就可以根据$$\alpha$$求出原始问题中的最优解$$w,b$$，使SVM问题得到解决。

SMO算法的基本思路是：如果当前所有变量的可行解都满足KKT条件，那么这个可行解就是最优解，因为KKT条件是该最优化问题的充要条件。否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题，这个二次规划关于这两个变量的解应该更接近原始二次规划问题的解。这样，SMO算法就分成了两个部分：求解一个二次规划问题和选择两个变量的方法。

为什么要选择2个变量？一般的，根据坐标上升算法的思想，我们会挑选一个变量出来，然后固定其他变量，求出这个变量的新的值，使得新值比原值更接近最优解。但是这里有一个约束条件$$\sum_{i=1}^{m} \alpha_{i}y^{(i)} = 0$$，如果我们固定$$m-1$$个变量，那么第$$m$$个变量的值也就固定了，因此，虽然我们挑选的是两个变量，但实际上只有一个变量是自由变量。

# 两个变量的二次规划的求解方法
不失一般性，我们假设选择的两个变量是$$\alpha_{1}, \alpha_{2}$$，其他的变量$$\alpha_{i}(i=3, 4, \cdots, m)$$都可以认为是不变的。

设：
```math
W(\alpha_{1}, \alpha_{2}) =  \frac{1}{2}\sum_{i,j=1}^{m}\alpha_{i}\alpha_{j}y^{(i)}y^{(j)}K(x^{(i)}, x^{(j)}) - \sum_{i=1}^{m}\alpha_{i}
```
```math
= \frac{1}{2}K_{11}\alpha_{1}^{2} + \frac{1}{2}K_{22}\alpha_{2}^{2} + K_{12}y^{(1)}y^{(2)}\alpha_{1}\alpha_{2} +  (y^{(1)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i1} - 1)\alpha_{1} + (y^{(2)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i2} - 1)\alpha_{2} + V_{constants}
```
其中$$K(x^{(i)}, x^{(j)})$$是核函数，$$K_{ij} = K(x^{(i)}, x^{(j)})$$, $$V_{constants}$$表示的是一些和$$\alpha_{1}, \alpha_{2}$$无关的常数项。

这样，我们把优化问题改写成：
```math
\min \limits_{\alpha_{1}, \alpha_{2}} W(\alpha_{1}, \alpha_{2})
```
```math
s.t. \quad y^{(1)}\alpha_{1} + y^{(2)}\alpha_{2} = -\sum_{i=3}^{m}y^{(i)}\alpha_{i} = \zeta
```
```math
0 \leqslant \alpha_{i} \leqslant C, i = 1, 2
```

首先分析约束条件，$$y^{(1)}\alpha_{1} + y^{(2)}\alpha_{2} = \zeta$$可以表示为2维空间中的一条直线，注意到$$y^{(1)}, y^{(1)}$$的值只能是$$\pm 1$$，所以直线的斜率是$$\pm 1$$，截距是$$\pm \zeta$$

在看不等式约束，这相当于把$$\alpha_{1}, \alpha_{2}$$的取值限制在一个边长为$$C$$的正方形内了。
如下图，当$$W(\alpha_{1}, \alpha_{2})$$的等值线和直线$$y^{(1)}\alpha_{1} + y^{(2)}\alpha_{2} = \zeta$$相切时，$$W(\alpha_{1}, \alpha_{2})$$取得最小值，但是，因为还有不等式约束，我们并不能保证切点在正方形内，这样，最小值就会出现在边界上，即需要对原来的解做裁剪
![image](http://oirw7ycq0.bkt.clouddn.com/smo_1.png)

我们把最优解记为$$\alpha_{1}^{new}, \alpha_{2}^{new}$$，初始化的可行解记为$$\alpha_{1}^{old}, \alpha_{2}^{old}$$，未经裁剪的最优解记为$$\alpha_{1}^{new, unc}, \alpha_{2}^{new, unc}$$

先不考虑裁剪，我们来求解$$\alpha_{1}^{new}, \alpha_{2}^{new}$$。
令$$g(x) = w^{T} + b = \sum_{i=1}^{m}\alpha_{i}y^{(i)}K(x^{(i)}, x) + b$$，记预测值和实际值之间的误差为$$E_{i} = g(x^{(i)}) - y^{(i)}$$
由等式约束条件
```math
y^{(1)}\alpha_{1} + y^{(2)}\alpha_{2} = \zeta
```
得：
```math
y^{(1)}\alpha_{1} = \zeta - y^{(2)}\alpha_{2}
```
等式两边同时乘上$$y^{(1)}$$
```math
\alpha_{1} = y^{(1)}(\zeta - y^{(2)}\alpha_{2})
```
代入到$$W(\alpha_{1}, \alpha_{2})$$中, 消去$$\alpha_{1}$$：
```math
W(\alpha_{2}) = \frac{1}{2}K_{11}(\zeta - y^{(2)}\alpha_{2})^{2} + \frac{1}{2}K_{22}\alpha_{2}^{2} + K_{12}y^{(2)}(\zeta - y^{(2)}\alpha_{2})\alpha_{2} +
```
```math
 (y^{(1)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i1} - 1)(\zeta - y^{(2)}\alpha_{2})y^{(1)} + (y^{(2)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i2} - 1)\alpha_{2} + V_{constants}
```
对$$\alpha_{2}$$求导数得：
```math
\frac{\mathrm{d} W(\alpha_{2})}{\mathrm{d} \alpha_{2}} = K_{11}\alpha_{2} + K_{22}\alpha_{2} - 2K_{12}\alpha_{2} - K_{11}y^{(2)}\zeta + K_{12}y^{(2)}\zeta - y^{(2)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i1}  + y^{(1)}y^{(2)} + y^{(2)}\sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i2} - 1
```
令$$\frac{\mathrm{d} W(\alpha_{2})}{\mathrm{d} \alpha_{2}} = 0$$得：
```math
(K_{11} + K_{12} -  2K_{12})\alpha_{2} = y^{(2)}(y^{(2)} - y^{(1)} + K_{11}\zeta - K_{12}\zeta + \sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i1} - \sum_{i=3}^{m}y^{(i)}\alpha_{i}K_{i2})
```
```math
= y^{(2)}(y^{(2)} - y^{(1)} + K_{11}\zeta - K_{12}\zeta + (g(x^{(1)}) - \sum_{i=1}^{2}y^{(i)}\alpha_{i}K_{i1} - b) - (g(x^{(2)}) - \sum_{i=1}^{2}y^{(i)}\alpha_{i}K_{i2} - b))
```
```math
= y^{(2)}(y^{(2)} - y^{(1)} + K_{11}\zeta - K_{12}\zeta + g(x^{(1)}) - g(x^{(2)}))
```
```math
= y^{(2)}(E_{1} - E_{2}) + y^{(2)}(K_{11} - K_{12})\zeta - (K_{11} - K_{12})y^{(1)}y^{(2)}\alpha_{1} + (K_{22} - K_{12})\alpha_{2}
```
再把$$\zeta = \alpha_{1}^{old}y^{(1)} + \alpha_{2}^{old}y^{(2)}$$代入上面等式中
```math
(K_{11} + K_{12} -  2K_{12})\alpha_{2}^{new, unc} = y^{(2)}(E_{1} - E_{2}) + y^{(2)}(K_{11} - K_{12})(\alpha_{1}^{old}y^{(1)} + \alpha_{2}^{old}y^{(2)}) - (K_{11} - K_{12})y^{(1)}y^{(2)}\alpha_{1}^{old} + (K_{22} - K_{12})\alpha_{2}^{old}
```
```math
= (K_{11} + K_{12} -  2K_{12})\alpha_{2}^{old} + y^{(2)}(E_{1} - E_{2})
```
所以有
```math
\alpha_{2}^{new, unc} = \alpha_{2}^{old} + \frac{y^{(2)}(E_{1} - E_{2})}{\eta}
```
其中
```math
\eta = K_{11} + K_{12} -  2K_{12} = \|\phi(x^{(1)}) - \phi(x^{(2)})\|^{2}
```
$$\phi(x)$$是输入空间到特征空间的映射。

下面再看如何来裁剪$$\alpha_{2}^{new, unc}$$
设$$\alpha_{2}^{new, unc}$$的上下界分别是$$L$$和$$H$$，
则有
```math
\alpha_{2}^{new} = \left\{\begin{matrix}
L, & \alpha_{2}^{new, unc} \lt L\\
\alpha_{2}^{new, unc}, & L \leqslant \alpha_{2}^{new, unc} \leqslant H \\
H, & \alpha_{2}^{new, unc} \gt H
\end{matrix}\right.
```
最后，再看$$L$$和$$H$$的取值，如下图
![image](http://oirw7ycq0.bkt.clouddn.com/smo_2.png)

当$$y^{(1)} = y^{(2)}$$时，如上面左图所示：
```math
L = \max(0, \alpha_{2}^{old} + \alpha_{1}^{old} - C), \quad H = \min(C, \alpha_{2}^{old} + \alpha_{1}^{old})
```
当$$y^{(1)} \neq y^{(2)}$$时，如上面右图所示：
```math
L = \max(0, \alpha_{2}^{old} - \alpha_{1}^{old}), \quad H = \min(C, C + \alpha_{2}^{old} - \alpha_{1}^{old})
```
# 变量选择的方法
## 第一个变量的选择
首先，再看KKT对偶互补条件：
```math
\alpha_{i}(1 - \xi_{i} - (w^{T}x^{(i)} + b)) = 0, i = 1, 2, \cdots, m
```
```math
\beta_{i}\xi_{i} = 0, i = 1, 2, \cdots, m
```
若$$\alpha_{i} = 0$$，则因为$$C = \alpha_{i} + \beta_{i}$$，所以由$$\beta_{i} = C \neq 0$$，则$$\xi_{i} = 0$$，可以得出：$$y^{(i)}g(x^{(i)}) \geqslant 1$$
若$$\alpha_{i} = C$$，则$$\beta_{i} = 0$$，所以$$\xi_{i} \geqslant 0$$，可以得出：$$y^{(i)}g(x^{(i)}) \leqslant 1$$
若$$0 \lt \alpha_{i} \lt C$$，则$$\beta_{i} \gt 0$$，所以$$\xi_{i} = 0$$，因为$$\alpha_{i} \neq 0$$，可以得出：$$y^{(i)}g(x^{(i)}) = 1$$
总结一下，KKT条件等价于：
```math
\alpha_{i} = 0 \quad \Leftrightarrow \quad y^{(i)}g(x^{(i)}) \geqslant 1
```
```math
0 \lt \alpha_{i} \lt C \quad \Leftrightarrow \quad y^{(i)}g(x^{(i)}) = 1
```
```math
\alpha_{i} = C \quad \Leftrightarrow \quad y^{(i)}g(x^{(i)}) \leqslant 1
```
SMO选择第一个变量的过程为外层循环，外层循环在训练样本中选取违反KKT条件最严重的样本点，并将其对应的$$\alpha_{i}$$作为第一个变量。
在检验过程中，首先遍历所有的支持向量点，即$$0 \lt \alpha_{i} \lt C$$的样点，如果这些点都满足KKT条件，再遍历整个训练集，看他们是否满足KKT条件。

## 第二个变量的选择
SMO选择第二个变量的过程为内层循环，假定已经找到第一个变量$$\alpha_{1}$$，那么找到第二个变量的标准是希望$$\alpha_{2}$$的变化能够足够大。
由上面的推导知，$$\alpha_{2}$$的变化依赖于$$|E_{1} - E_{2}|$$，一种简单的做法就是选择$$\alpha_{2}$$，使得$$|E_{1} - E_{2}|$$最大。但是，如果$$\eta$$为零，也就是说，如果由两个变量的特征值相同，就会出现$$\eta$$为零的情况，这时，需要重新选择第二个变量。具体的方法是：先遍历支持向量点，如果找不到合适的$$\alpha_{2}$$，再遍历整个集合，如果也找不到合适的$$\alpha_{2}$$，就放弃已经选择的第一个变量。

# 阈值b和误差E的计算
每次完成两个变量的优化后，也要对$$b$$的值进行更新，因为$$b$$的值关系到下一次优化时$$E_{i}$$的计算。
当$$\alpha_{1}^{new}$$不在边界上时，由KKT条件知：$$y^{(1)}g(x^{(1)}) = 1$$，即
```math
\sum_{i=1}^{m} \alpha_{1}y^{(i)}K_{i1} + b = y^{(1)}
```
于是
```math
b_{1}^{new} = y^{(1)} - \sum_{i=3}^{m} \alpha_{1}y^{(i)}K_{i1} - \alpha_{1}^{new}y^{(1)}K_{11} - \alpha_{2}^{new}y^{(2)}K_{21}
```
又因为
```math
E_{1}^{old} = \sum_{i=3}^{m} \alpha_{1}y^{(i)}K_{i1} + \alpha_{1}^{old}y^{(1)}K_{11} + \alpha_{2}^{old}y^{(2)}K_{21}  + b^{old} - y^{(1)}
```
两式相加：
```math
b_{1}^{new} + E_{1}^{old} = \alpha_{1}^{old}y^{(1)}K_{11} + \alpha_{2}^{old}y^{(2)}K_{21} - \alpha_{1}^{new}y^{(1)}K_{11} - \alpha_{2}^{new}y^{(2)}K_{21} + b^{old}
```
即
```math
b_{1}^{new} = b^{old} - (E_{1}^{old} + y^{(1)}K_{11}(\alpha_{1}^{new} - \alpha_{1}^{old}) + y^{(2)}K_{21}(\alpha_{2}^{new} - \alpha_{2}^{old}))
```
同样，当$$\alpha_{2}^{new}$$不在边界上时，有：
```math
b_{2}^{new} = b^{old} - (E_{2}^{old} + y^{(1)}K_{12}(\alpha_{1}^{new} - \alpha_{1}^{old}) + y^{(2)}K_{22}(\alpha_{2}^{new} - \alpha_{2}^{old}))
```

如果$$\alpha_{1}^{new}, \alpha_{2}^{new}$$同时满足条件$$0 \lt \alpha_{i}^{new} \lt C, i=1,2$$，那么有$$b^{new} = b_{1}^{new} = b_{2}^{new}$$，若$$\alpha_{1}^{new}, \alpha_{2}^{new}$$是$$0$$或$$C$$，那么$$b_{1}^{new}, b_{2}^{new}$$以及他们之间的值都是符合KKT条件的阈值，我们可以选择$$b^{new} = \frac{b_{1}^{new} + b_{2}^{new}}{2}$$。

每次更新完两个变量的值后，还需要更新其对应的误差$$E_{i}$$，
```math
E_{i}^{new} = \sum_{j=1}^{m} \alpha_{j}y^{(j)}K_{ij} - y^{(i)}
```
# SMO算法流程
输入：样本集$${(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})}$$，其中，$$x^{(i)} \in R^{n}, y^{(i)} \in {1, -1}, i = 1, 2, \cdots, m$$，精度$$\varepsilon$$
输出：近似解$$\hat{\alpha}$$
算法流程：
(1) 令迭代变量$$k=0$$，取$$\alpha^{(0)} = 0$$
(2) 选取优化变量$$\alpha_{1}^{(k)}, \alpha_{2}^{(k)}$$，求解两个变量的最优化问题，得到$$\alpha_{1}^{(k+1)}, \alpha_{2}^{(k+1)}$$，将$$\alpha^{k}$$更新为$$\alpha^{k+1}$$
(3) 若在精度$$\varepsilon$$的范围内满足下列条件：
```math
\sum_{i=1}^{m} \alpha_{i}y^{(i)} = 0
```
```math
0 \leqslant \alpha_{i} \leqslant C, i = 1,2, \cdots, m
```
```math
y^{(i)}(\sum_{j=1}^{m}\alpha_{j}y^{(j)}K_{ij} + b) \left\{\begin{matrix}
 \geqslant 1, & \alpha_{i} = 0\\
 =1, & 0 \lt \alpha_{i} \lt  C \\
 \leqslant 1, & \alpha_{i} = C
\end{matrix}\right. \quad i = 1,2, \cdots, m
```
则，转(4)，否则令$$k=k+1$$，转(2)
(4) 取$$\hat{\alpha} = \alpha^{k+1}$$，算法结束
