# <center> 概率算法作业 </center>
<p align="right">From: USTC研究生课程《算法设计与分析》</p>
<p align="right">By: SA21011259 张号</p>

- **P20-EX1:  $\pi$的近似值计算中(P19)，若将$y\gets uniform(0,1)$改为$y\gets x$，则算法的估计值为多少？**
解：由于将$y\gets uniform(0,1)$改为$y\gets x$，则生成的点将均匀落在$y=x$，$x\in [0,\frac{\sqrt{2}}{2}]$这条直线上，所以$\frac{4k}{n}\sim2\sqrt{2}$。
<p> 

- **P23-EX2: 在机器上用$4\int_0^1\sqrt{1-x^2}dx$估计$\pi$值，给出不同的$n$值及精度。**
解：代码如下：
    ```
    import math
    f = lambda x:math.sqrt(1-x**2)
    slices = [1000, 10000, 100000, 100000]
    ans = []
    for slice in slices:
        sum = 0
        dx = 1.0 / slice
        for i in range(slice):
            sum += f(i * dx) * dx
        ans.append(sum * 4)
    print(ans)
    ```
    <div class="center">

    |  分割片数 |    结果   |  精度   |
    |:--------:|:---------:|:-------:|
    |   1,000  |3.143555466911028|0.002|
    |  10,000  |3.1417914776113167|0.0002|
    | 100,000  |3.1416126164019564|0.00002|
    |1,000,000 |3.1415946524138207|0.000002|
    </div>
    
- **P23-EX3: 设$a$，$b$，$c$和$d$是实数，且$a≤b$，$c≤d$，$f:[a, b] → [c,d]$是一个连续函数，写一概率算法计算积分: $\int_b^af(x)dx$
    \*注意，函数的参数是$a$，$b$，$c$，$d$，$n$和$f$, 其中$f$用函数指针实现，请选一连续函数做实验，并给出实验结果。**
解：令$f(x)=x^3-2$,所求积分f:[-1, 2] → [-3, 6]，算法求得值为-2.246265，真实值为-2.5。以下为代码：
    ```
    import numpy as np
    def F(a, b, c, d, n, f):
        assert a <= b and c <= d and n > 0
        sum = 0
        for i in range(n):
            x = np.random.uniform(a,b)
            y = np.random.uniform(c,d)
            if f(x) > y and y >= 0.:
                sum += 1
            elif f(x) <= y and y < 0.:
                sum -= 1
        return float(sum) / n * (b-a) * (d - c)
    f = lambda x:x**3-2
    print (F(-1,2,-3,6,1000000,f)) 
    ```
    
- **P24-EX4: 设$ε$，$δ$是(0,1)之间的常数，证明：
	    若$I$是$\int_0^1f(x)dx$的正确值，$h$是由HitorMiss算法返回的值，则当$n ≥ I(1-I)/ε^2δ$时有：$$Prob[|h-I| < ε] ≥ 1 – δ$$
        上述的意义告诉我们：$Prob[|h-I| ≥ ε] ≤δ$, 即：当$n ≥ I(1-I)/ε^2δ$时，算法的计算结果的绝对误差超过$ε$的概率不超过$δ$，因此我们根据给定$ε$和$δ$可以确定算法迭代的次数。
		解此问题时可用切比雪夫不等式，将$I$看作是数学期望。**
证明：
- **P36-EX5**: 用P35算法，估计整数子集$1\sim n$的大小，并分析$n$对估计值的影响。
- **P54-EX6**: 分析dlogRH的工作原理，指出该算法相应的$u$和$v$
- **P67-EX7**: 写一Sherwood算法C，与算法A, B, D比较，给出实验结果。
- **P77-EX8**: 证明：当放置$(k+1)_{th}$皇后时，若有多个位置是开放的,则算法QueensLV选中其中任一位置的概率相等。
- **P83-EX9**: 写一算法，求n=12~20时最优的StepVegas值。


