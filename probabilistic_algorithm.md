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
解：令$f(x)=x^3-2$,所求积分f:[-1, 2] → [-3, 6]，算法求得值为-2.246265，真实值为-2.25。以下为代码：
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
证明：$k=\frac{nh}{S_D}$为进行n次重复独立实验中命中的次数，每次命中的概率为$p=\frac{I}{S_D}$，$S_D$为投掷范围总面积。令$X=h$，则$X\sim B(S_D,p)$，期望$u=I$，方差$D(X)=\frac{I(S_D-I)}{S_D}$，根据切比雪夫不等式得：$$Prob[|h-I| < ε] ≥ 1 – δ$$
其中$ε$为给定误差，$δ=\frac{I(S_D-I)}{ε^2S_D}$
<p>

- **P36-EX5: 用P35算法，估计整数子集$1\sim n$的大小，并分析$n$对估计值的影响。**
解：代码如下：
    ```
    def getK(X):
        S = []
        k = 0
        getUniformX = lambda :random.sample(X, 1)[0]
        a = getUniformX()
        while True:
            k += 1
            S.append(a)
            a = getUniformX()
            if a in S:
                break
        return k

    def SetCount(X, times=1000):
        ave = 0.
        for _ in range(times):
            ave += getK(X)
        ave /= times
        return 2 * ave * ave / math.pi

    N = [5, 10, 50, 100, 500, 1000]
    ans = []
    for n in N:
        X = list(np.arange(1, n+1))
        ans.append(SetCount(X))
    print(ans)
    ```
    <div class="center">

    |     $n$   |    估计值   |
    |:--------:|:---------:|
    |   1  |0.637|
    |   5  |4.033|
    |  10  |8.744|
    | 50  |45.252|
    |100 |96.173|
    |500 |498.825|
    |1,000 |952.326|
    |5,000 |5,043.345|
    |10,000 |10044.185|
    </div>

    从表中可以看出，当$n$取值较小时，概率算法所得结果与真实值偏差较大，随着$n$的增大，其估计值越来越接近真实值。
<p>   

- **P54-EX6: 分析dlogRH的工作原理，指出该算法相应的$u$和$v$
定理1：$\log_{g,p}(st\,mod\,p)=(\log_{g,p}s+\log_{g,p}t)\,mod\,(p-1)$
定理2：$\log_{g,p}(g^r\,mod\,p)=r\quad for\quad 0≤r≤p-2 $**
解：$r=uniform(0..p-2)$，随机实例$c=u(g,r,p,a)=[a(g^r\,mod\,p)]\,mod\,p$，调用确定性算法得到解$y=f(g,p,c)=\log_{g,p}c$，最后还原为原本解$x=v(r,y)=(y-r)\,mod\,(p-1)$。
原理：$x=(y-r)\,mod\,(p-1)=y\,mod\,(p-1)-r\,mod\,(p-1)$，其中$y\,mod\,(p-1)=\log_{g,p}c\,mod\,(p-1)=\log_{g,p}((a(g^r\,mod\,p)))\,mod\,p\,mod\,(p-1)$，由因为$a=g^x\,mod\,p$和**定理1**，所以$\log_{g,p}(a(g^r\,mod\,p))=\log_{g,p}(g^{r+x}\,mod\,p)=(\log_{g,p}g^r+\log_{g,p}g^x)\,mod\,(p-1)$，由**定理2**得，上式$=(r+x)\,mod\,(p-1)$，所以$(y-r)\,mod\,(p-1)=x\,mod\,(p-1)$，又因为$x\ne p-1$，所以上述算法成立。

<p>

- **P67-EX7: 写一Sherwood算法C，与算法A, B, D比较，给出实验结果。**
  解：定义有序链表结构如下：
  ```
  class OrderedList():
      def __init__(self, len=10000) -> None:
          self.len = len
          self.val = []
          self.next = []
          self.pre = []
          #生成测试数据
          for i in range(len):
              self.val.append(i)
              self.next.append(i+1)
              self.pre.append(i-1)
          self.next[-1] = 0
          self.pre[0] = len - 1
          self.head = 0
          #打乱数据
          self.shuffle()

      def shuffle(self):
          for i in range(self.len):
              r = np.random.randint(self.len)
              if self.head == i:
                  self.head = r
              elif self.head == r:
                  self.head = i

              self.val[i], self.val[r] = self.val[r], self.val[i]

              self.next[self.pre[i]], self.next[self.pre[r]] = r, i
              self.pre[i], self.pre[r] = self.pre[r], self.pre[i]
              self.pre[self.next[i]], self.pre[self.next[r]] = r, i
              self.next[i], self.next[r] = self.next[r], self.next[i]

      #serach函数，从ptr位置开始查找，返回可能对应x的ptr和查找计数
      def __call__(self, x, ptr=None):
          if ptr is None:
              ptr = self.head
          assert ptr < self.len
          cnt = 1
          while x > self.val[ptr]:
              cnt += 1
              ptr = self.next[ptr]
              assert ptr != self.head
          return ptr, cnt

      def getRandomPtr(self):
          return np.random.randint(self.len)
  ```
  算法A：
  ```
  def A(x, ordered_list):
      return ordered_list(x)
  ```
  算法B：
  ```
  def B(x, ordered_list):
      ptr  = ordered_list.head
      _max = ordered_list.val[ptr]
      batch = int(math.sqrt(ordered_list.len))

      for i in range(batch):
          y = ordered_list.val[i]
          if _max < y and y <= x:
              ptr = i
              _max = y

      return ordered_list(x, ptr)
  ```
  算法C：
  ```
  def C(x, ordered_list):
      ptr  = ordered_list.head
      _max = ordered_list.val[ptr]
      batch = int(math.sqrt(ordered_list.len))
      choices = np.random.choice(ordered_list.len, size=batch, replace=False)

      for choice in choices:
          y = ordered_list.val[choice]
          if _max < y and y <= x:
              ptr = choice
              _max = y

      return ordered_list(x, ptr)
  ```
   算法D：
  ```
  def D(x, ordered_list):
      ptr = ordered_list.getRandomPtr()
      y = ordered_list.val[ptr]
      if y > x:
          ptr = None
      elif y < x:
          ptr = ordered_list.next[ptr]
      else:
          return ptr, 0
      return ordered_list(x, ptr)
  ```
  测试程序：
  ```
  len = 10000
  test = OrderedList(len)
  test_times = 5000

  worst = {"A":0, "B":0, "C":0, "D":0}
  average = {"A":0., "B":0., "C":0., "D":0.}
  alg_s = {"A":A, "B":B, "C":C, "D":D}

  for name, alg in alg_s.items():
      for i in range(test_times):
          x = np.random.randint(len)
          ptr, cnt = alg(x, test)
          assert test.val[ptr] == x
          average[name] += cnt
          if cnt > worst[name]:
              worst[name] = cnt
      average[name] /= test_times

  print(worst)
  print(average)
  ```
  测试结果为：
    创建了长度为10,000的有序链表，即$n=10000$，创建完进行打乱。分别测试了四个算法的最坏情况和平均情况。表中记录为查询5000次对应算法的单次查询中最坏访问链表元素次数和平均访问次数。
   <div class="center">

    |算法|A|B|C|D|     
    |:--------:|:---------:|:---------:|:---------:|:---------:|
    |最坏情况|10,000|483|892|9,785|
    |平均情况|4,950.70|91.28|98.31|3,286.06|

    </div>

    以上结果符合相应算法的时间复杂度。综合而言B和C算法的平均性能和最差性能皆优于A和D，B和C的平均时间复杂度皆为O$(\sqrt{n})$，A的最坏复杂度为O$(n)$，平均复杂度为O$(\frac{n}{2})$，A的最坏复杂度为O$(n-1)\approx O(n)$，平均复杂度为O$(\frac{n}{3})$。值得注意的是，经过Sherwood处理后的C算法最坏性能并未优于B算法。分析其主要原因为该链表在初始过程中，已经经过一次随机打乱操作，链表物理顺序存储的前$\sqrt{n}$个元素已是随机元素。

<p>


- **P77-EX8: 证明：当放置$(k+1)_{th}$皇后时，若有多个位置是开放的，则算法QueensLV选中其中任一位置的概率相等。**
  
  解：当放置$(k+1)_{th}$皇后时，有$n$个位置开放，即证每个位置被选中的概率皆为$\frac{1}{n}$。
  1).假设循环到第$m(m\in [1,n-1])$个位置时选中最后一个位置，剩余未访问的位置的在将来访问时皆未被选中，第$m$轮选中该位置的概率为$\frac{1}{m}$，根据乘法原则，剩余位置未被选择的概率为$$\prod_{i=m+1}^{n}\frac{i-1}{i}=\frac{m}{n}$$所以最终选择第$m$个位置的概率为$\frac{1}{m}\times \frac{m}{n}=\frac{1}{n}$。
  2).当选中最后一个位置时，其概率为$\frac{1}{n}$。
  综上，选中任何一个位置的概率皆为$\frac{1}{n}$，即选中每个位置概率相等。

<p>

- **P83-EX9: 写一算法，求n=12~20时最优的StepVegas值。**
  解：$n$皇后的回溯算法如下：
  ```
  def backtrace(target_k, col, diag45, diag135, row=0):
      if row == target_k:
          return True, 0
        
      cnt = 0
      for _col in range(target_k):
          if _col not in col and _col - row not in diag45 \
              and _col + row not in diag135:
              col.append(_col)
              diag45.append(_col - row)
              diag135.append(_col + row)
              #访问该节点，对应计数增1 
              cnt += 1
              isSuccess, _cnt = backtrace(target_k, col, diag45
                                          , diag135, row+1)
              #增加对应子节点的计数
              cnt += _cnt
              if isSuccess:
                  return True, cnt
              col.pop()
              diag45.pop()
              diag135.pop()

      return False, cnt
  ```
  贪心的LV算法，前StepVegas个皇后随机放置，代码如下：
  ```
  def QueensLV(target_k, stepVegas):
    col = []
    diag45 =[]
    diag135 = []
    row = 0
    k = 0
    nb = 0

    while True:
        nb = 0
        for _col in range(target_k):
            if _col not in col and _col - row not in diag45 \
                and _col + row not in diag135:
                nb += 1
                if np.random.randint(nb) == 0:
                    col_candidate = _col
        if nb > 0:
            col.append(col_candidate)
            diag45.append(col_candidate - k)
            diag135.append(col_candidate + k)
            k += 1
        if nb == 0 or k >= stepVegas:
            break
    
    if nb > 0:
        isSuccess, cnt = backtrace(target_k, col, diag45, diag135, k)
        return isSuccess, cnt + stepVegas
    else:
        return False, stepVegas
  ```
  以下为测试代码：
  
  ```
  times = 1000
  epoch = 100
  ans = {}
  for n in range(12, 21):
      best_s = 0.
      best_e = 0.
      best_t = math.inf
      bestStepVeags = 0
      bestSuccessRate = 0.
      for stepVegas in range(1, n + 1):
          successRate = 0.
          s = 0.
          e = 0.

          for _ in range(times):
              isSuccess, cnt = QueensLV(n, stepVegas)
              if isSuccess:
                  successRate += 1
                  s += cnt
              else:
                  e += cnt
        
          successRate += 1e-6
          s /= successRate
          e /= times - successRate + 1e-6

          successRate /= times
        
          t = s + (1 - successRate) * e / successRate
          if t < best_t:
              best_t = t
              best_e = e
              best_s = s
              bestSuccessRate = successRate
              bestStepVeags = stepVegas
      ans[n] = (bestStepVeags, bestSuccessRate, best_s, best_e, best_t)

  print(ans)
  ```
  以下为对应结果：
    
  <div class="center">

  |$n$|best StepVegas|$p$|$s$|$e$|$t$|     
  |:-----:|:-------:|:------:|:-----:|:-----:|:----:|
  |12|6|0.535|22.415|25.335|44.436|
  |13|7|0.495|22.341|24.590|47.428|
  |14|8|0.461|23.714|23.122|50.748|
  |15|9|0.425|24.873|22.769|55.678|
  |16|10|0.371|25.431|21.620|62.086|
  |17|10|0.563|36.710|39.091|67.053|
  |18|11|0.485|38.800|35.087|76.057|
  |19|12|0.405|38.462|30.708|83.575|
  |20|13|0.358|39.263|25.774|85.483|

  </div>
