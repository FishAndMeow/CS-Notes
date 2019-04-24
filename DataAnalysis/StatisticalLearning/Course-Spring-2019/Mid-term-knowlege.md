# 统计机器学习 期中

**知识点总结**

朴素贝叶斯分类 $p(x, y)=p(y) p(x | y)$ 并不一定是线性，而如果$p(x_i|c)$为指数族分布，则为非线性

## Supervised Learning  —— Classification

Hard SVM 

<<<<<<< HEAD
原始的优化问题为：
$$
\begin{array}{ll}{\min _{w, b}} & {\frac{1}{2}\|w\|^{2}} \\ {\text { s.t. }} & {y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$
分离超平面为：
$$
w^{\star}x + b^{\star} = 0
$$
分类决策函数为：
$$
f(x) = sign (w^{\star}x + b^{\star})
$$
线性可分支持向量机的最优解存在且唯一，超平面由支持向量完全决定。

二次规划的对偶问题是
$$
\min \quad \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$

$$
\begin{array}{c}{\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ {\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$

引入松弛变量$\xi_i$，原始最优化问题为
$$
\min _{m, b . \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
$$

$$
\begin{array}{c}{\text { s.t. } \quad y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N} \\ {\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$

此时解$w^{\star}$唯一但$b^{\star}$不唯一

对偶问题为
$$
\min _{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$

$$
\begin{aligned} \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N \end{aligned}
$$

对于非线性情况，有
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x, x_{i}\right)+b^{*}\right)
$$
其中对称函数
$$
K(x, z)=\phi(x) \cdot \phi(z)
$$
为正定核的充要条件为：对任意$x_{i} \in \mathcal{X}, i=1,2, \cdots, m$，任意正整数 $m$， 其对应的 Gramm 矩阵是**半正定**的

+ If the data is not linearly separable, adding new features might make it linearly separable 
+ Soft margin SVM instead of hard SVM 

支持向量是使得下述约束等号成立的点：
$$
y_i(w x_i+b)-1 = 0
$$
对于 $y_i=1$的点，支持向量在超平面 $H_1:wx+b=1$上，对于$y_i=-1$的点，支持向量在超平面$H_2:wx+b=-1$上 

训练数据中对应 $\alpha^{\star}>0$的点才会对结果产生影响，这些点就是支持向量

=======
+ If the data is not linearly separable, adding new features might make it linearly separable 
+ Soft margin SVM instead of hard SVM 

>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
Approximation of the step function 

+ 0-1 Loss
+ Quadratic loss
+ hinge loss (**upper bounds the 0/1 loss**)

Approximation of 0/1 loss

+ Piecewise linear approximation(hinge loss) 
+ Quadratic approximation(square-loss)
+ Huber loss(combine the above two)

<<<<<<< HEAD
![1](C:\Users\feiyuxiao\Desktop\1.png) 

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
**The Primal Soft SVM problem**
$$
\hat{\mathbf{w}}=\arg \min _{\mathbf{w}} \frac{\lambda}{2}\|\mathbf{w}\|^{2}+\sum_{i=1}^{N} \xi_{i}
$$
where  $\xi_{i} :=\ell_{l i n}\left(y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}\right)=\max \left(0,1-y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}\right)$

等价地有
$$
\begin{aligned} \hat{\mathbf{w}}=\arg \min _{\mathbf{w}, \xi} \frac{\lambda}{2}\|\mathbf{w}\|^{2} &+\sum_{i=1}^{N} \xi_{i} \\ \text { s.t.: } y_{i} \mathbf{w}^{\top} \mathbf{x}_{i} & \geq 1-\xi_{i}, \forall i=1, \ldots, N \\ & \quad \xi_{i} \geq 0, \forall i=1, \ldots, N \end{aligned}
$$
变形为
$$
\begin{array}{c}{\hat{\mathbf{w}}=\arg \min _{\mathbf{w}, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^{2}+C \sum_{i=1}^{N} \xi_{i}} \\ {C=\frac{1}{\lambda}}\end{array}
$$
Lagrange multipliers
$$
\alpha \geq 0, \beta \geq 0
$$
Lagrangian function
$$
L(\mathbf{w}, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta})=\frac{1}{2}\|\mathbf{w}\|^{2}+C \sum_{i} \xi_{i}-\sum_{i} \alpha_{i}\left(y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}-1+\xi_{i}\right)-\sum_{i} \beta_{i} \xi_{i}
$$

$$
\min _{\mathbf{w}, \boldsymbol{\xi}} \max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} L(\mathbf{w}, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta})
$$

写成向量形式有
$$
L(\mathbf{w}, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta})=\frac{1}{2}\|\mathbf{w}\|^{2}+C \boldsymbol{\xi}^{\top} \mathbf{1}-\sum_{i} \alpha_{i} y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}+\boldsymbol{\alpha}^{\top} \mathbf{1}-\boldsymbol{\xi}^{\top}(\boldsymbol{\alpha}+\boldsymbol{\beta})
$$
求导
$$
\begin{array}{l}{0=\left.\frac{\partial L}{\partial \mathbf{w}}\right|_{\hat{\mathbf{w}}} \quad \Rightarrow \hat{\mathbf{w}}=\sum_{i=1}^{N} \alpha_{i} y_{i} \mathbf{x}_{i}} \\ {0=\left.\frac{\partial L}{\partial \boldsymbol{\xi}}\right|_{\hat{\xi}} \quad \Rightarrow \boldsymbol{\beta}=C 1-\boldsymbol{\alpha} \geq 0} \\ {\Rightarrow 0 \leq \boldsymbol{\alpha} \leq C \mathbf{1}}\end{array}
$$
有
$$
\begin{array}{l}{(\hat{\boldsymbol{\alpha}}, \hat{\boldsymbol{\beta}})=\underset{0 \leq \boldsymbol{\alpha} \leq C \mathbf{1} ; 0 \leq \boldsymbol{\beta}}{\operatorname{argmax}} L(\hat{\mathbf{w}}, \hat{\boldsymbol{\xi}}, \boldsymbol{\alpha}, \boldsymbol{\beta})} \\ {\hat{\boldsymbol{\alpha}}=\underset{0 \leq \boldsymbol{\alpha} \leq C \mathbf{1}}{\operatorname{argmax}} \boldsymbol{\alpha}^{\top} \mathbf{1}-\frac{1}{2} \boldsymbol{\alpha}^{\top} \boldsymbol{Y} G \mathbf{Y} \boldsymbol{\alpha}}\end{array}
$$
定义
$$
\mathbf{Y} :=\operatorname{diag}\left(y_{1}, \ldots, y_{N}\right)
$$
和 Gram matrix $G \in \mathbb{R}^{N \times N}$, where $G_{i j} :=\mathrm{x}_{i}^{\top} \mathrm{x}_{j}$
$$
\hat{\boldsymbol{\alpha}}=\underset{0 \leq \alpha \leq C 1}{\operatorname{argmax}} \boldsymbol{\alpha}^{\top} \mathbf{1}-\frac{1}{2} \boldsymbol{\alpha}^{\top} \mathbf{Y} G \mathbf{Y} \boldsymbol{\alpha}
$$
代入
$$
\hat{\mathbf{w}}=\sum_{i=1}^{N} \hat{\alpha}_{i} y_{i} \mathbf{x}_{i}
$$
有
$$
f(\mathbf{x} ; \hat{\mathbf{w}})=\operatorname{sign}\left(\hat{\mathbf{w}}^{\top} \mathbf{x}\right)=\operatorname{sign}\left(\sum_{i=1}^{N} \hat{\alpha}_{i} y_{i} \mathbf{x}_{i}^{\top} \mathbf{x}\right)
$$
**Support vectors in Soft SVM**

+ Margin support vectors
  $$
  y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}=1
  $$

+ Non-margin support vectors
  $$
  \xi_{i}>0
  $$

Only few Lagrange multipliers (dual variables) $\alpha_i$ can be non-zero 

**For multiple classes**

Learn multiple binary classifiers separately

+ The weights may not be based on the same scale 
+ Imbalance issue when learning each binary classifier 

Learning one joint classifier
$$
\begin{array}{c}{\min _{\mathbf{w}, b} \frac{1}{2} \sum_{y}\left\|\mathbf{w}_{y}\right\|^{2}+C \sum_{i=1}^{N} \sum_{y \neq y_{i}} \xi_{i y}} \\ {\text { s.t. } : \mathbf{w}_{y_{i}}^{\top} \mathbf{x}_{i}+b_{y_{i}} \geq \mathbf{w}_{y}^{\top} \mathbf{x}_{i}+b_{y}+1-\xi_{i y} \forall i, \forall y \neq y_{i}} \\ {\xi_{i y} \geq 0 \quad \quad \forall i, \forall y \neq y_{i}}\end{array}
$$
Prediction
$$
\hat{y}=\underset{k}{\operatorname{argmax}}\left(\mathbf{w}_{k}^{\top} \mathbf{x}+b_{k}\right)
$$
**Non-linear SVMs: Feature Spaces**

the original feature space can always be mapped to some higher-dimensional feature space where the training set is separable 

+ A kernel function is a function that is equivalent to an inner product in some feature space 
+ Every semi-positive definite symmetric function is a kernel 
+ Semi-positive definite symmetric functions correspond to a semi-positive definite symmetric Gram matrix 

**Holdout Method**

Split dataset into two subsets 

+ Training set: used to learn the classifier 
+ Test set: used to estimate the error rate of the trained classifier 

Resampling

+ Cross-validation : Random sub-sampling ,K-fold cross-validation , Leave-one-out cross-validation 
+ Bootstrap

## Probabilistic Methods for Classification 

**Naive Bayes Classifier**

<<<<<<< HEAD
朴素贝叶斯分类是典型的**生成学习方法**，其基本假设是**条件独立性** 
$$
\begin{aligned} P(X&=x | Y=c_{k} )=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\ &=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right) \end{aligned}
$$

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
+ Naive Bayes assumption : features $X_{1}, \ldots, X_{d}$ are conditionally independent given the class label $Y$

A joint distribution:  $p(\mathrm{x}, y)=p(y) p(\mathrm{x} | y)$

其中 $p(y)$ 为 **prior** , $p(x|y)$ 为 likelihood

Bayes Rule:
$$
p(y | \mathrm{x})=\frac{p(\mathrm{x}, y)}{p(\mathrm{x})}=\frac{p(y) p(\mathrm{x} | y)}{p(\mathrm{x})}
$$
Bayes's decision rule:
$$
y^{*}=\arg \max _{y \in \mathcal{Y}} p(y | \mathrm{x})
$$
**Maximum likelihood estimation**
$$
p\left(\left\{\mathbf{x}_{i}, y_{i} | \pi, q\right\}\right)=\prod_{i=1}^{N} p\left(\mathbf{x}_{i}, y_{i} | \pi, q\right)
$$

$$
\begin{array}{l}{(\hat{\pi}, \hat{q})=\arg \max _{\pi, q} p\left(\left\{\mathbf{x}_{i}, y_{i}\right\} | \pi, q\right)} \\ {(\hat{\pi}, \hat{q})=\arg \max _{\pi, q} \log p\left(\left\{\mathbf{x}_{i}, y_{i}\right\} | \pi, q\right)}\end{array}
$$

Results
$$
\begin{array}{c}{\hat{\pi}=\frac{N_{1}}{N} \quad \hat{q}_{0 j}=\frac{N_{0}^{j}}{N_{0}} \quad \hat{q}_{1 j}=\frac{N_{1}^{j}}{N_{1}}} \\ {N_{k}=\sum_{i=1}^{N} \mathbf{I}\left(y_{i}=k\right) : \# \text { of data in category } k} \\ {N_{k}^{j}=\sum_{i=1}^{N} \mathbf{I}\left(y_{i}=k, x_{i j}=1\right) : \# \text { of data in category } k \text { that has feature } j}\end{array}
$$
Data scarcity issue: **Laplace smoothing (Additive smoothing) **
$$
\begin{aligned} \hat{q}_{0 j}=& \frac{N_{0}^{j}+\alpha}{N_{0}+2 \alpha} \\ \hat{q}_{1 j} &=\frac{N_{1}^{j}+\alpha}{N_{1}+2 \alpha} \end{aligned}
$$
$\alpha > 0$

<<<<<<< HEAD
**A Bayesian Treatment**

Put a prior on the parameters
$$
p_{0}\left(q_{0 j} | \alpha_{1}, \alpha_{2}\right)=\operatorname{Beta}\left(\alpha_{1}, \alpha_{2}\right)=\frac{\Gamma\left(\alpha_{1}+\alpha_{2}\right)}{\Gamma\left(\alpha_{1}\right) \Gamma\left(\alpha_{2}\right)} q_{0 j}^{\alpha_{1}-1}\left(1-q_{0 j}\right)^{\alpha_{2}-1}p_{0}\left(q_{0 j} | \alpha_{1}, \alpha_{2}\right)=\operatorname{Beta}\left(\alpha_{1}, \alpha_{2}\right)=\frac{\Gamma\left(\alpha_{1}+\alpha_{2}\right)}{\Gamma\left(\alpha_{1}\right) \Gamma\left(\alpha_{2}\right)} q_{0 j}^{\alpha_{1}-1}\left(1-q_{0 j}\right)^{\alpha_{2}-1}
$$
**Maximum a Posterior Estimate (MAP)**
$$
\begin{aligned} \hat{q} &=\arg \max _{q} \log p\left(q |\left\{\mathbf{x}_{i}, y_{i}\right\}\right) \\ &=\arg \max _{q} \log p_{0}(q)+\log p\left(\left\{\mathbf{x}_{i}, y_{i}\right\} | q\right) \end{aligned}
$$

$$
\hat{q}_{0 j}=\frac{N_{0}^{j}+\alpha_{1}-1}{N_{0}+\alpha_{1}+\alpha_{2}-2}
$$

If $\alpha_1 = \alpha_2 = 1$, no effect:

**MLE is a special case of Bayesian estimate** 

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
**Gaussian Naive Bayes**(GNB)

Different mean and variance for each class k and each feature i 

+ decision boundary of NB
  + GNB with equal variance  leads to a linear decision boundary
+ For multi-class, the predictive distribution is softmax! 

**Generative vs. Discriminative Classifiers **

Generative classifiers (example: Naive Bayes)

+ Assume some functional form for  $P(X,Y)$(or $P(Y)$ and $P(X|Y)$)

+ Estimate parameters of $P(X,Y)$ directly from training data

+ Make prediction
  $$
  \hat{y}=\underset{y}{\operatorname{argmax}} P(\mathbf{x}, Y=y)
  $$

Discriminative classifiers  (Example: Logistic regression)

+ Assume some functional form for  $P(Y|X)$
+ Estimate parameters of  $p(Y|X)$ directly from training data

<<<<<<< HEAD
对于生成方法，可以还原出联合概率分布，其学习收敛速率更快，即当样本容量增加的时候，模型可以更快地收敛于真实模型，当存在隐变量时，依然可以用生成方法学习；对于判别方法，其直接学习的是条件概率或者决策函数，往往其准确率更高，可以对数据进行各种程度上的抽象，定义特征并使用特征，从而可以简化学习问题。

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
### Logistic Regression

Assume
$$
P(y=1 | \mathbf{x})=\frac{1}{1+\exp \left(-\left(w_{0}+\mathbf{w}^{\top} \mathbf{x}\right)\right)}
$$
**Logistic regression is a linear classifier**

Multi-class Logistic Regression : piecewise linear

Solving
$$
\hat{\mathbf{w}}=\underset{\mathbf{w}}{\operatorname{argmax}} \prod_{i=1}^{N} P\left(y_{i} | \mathbf{x}_{i}, \mathbf{w}\right)
$$
We have
$$
\begin{aligned} \mathcal{L}(\mathbf{w}) &=\log \prod_{i=1}^{N} P\left(y_{i} | \mathbf{x}_{i}, \mathbf{w}\right) \\ &=\sum_{i}\left[y_{i} \mathbf{w}^{\top} \mathbf{x}_{i}-\log \left(1+\exp \left(\mathbf{w}^{\top} \mathbf{x}_{i}\right)\right)\right] \end{aligned}
$$

+ no closed-form solution
+ concave function of $w$

GNB with class-independent variances representationally equivalent to LR 

### Exponential family

$$
\begin{aligned} p(\mathbf{x} | \boldsymbol{\eta}) &=h(\mathbf{x}) \exp \left(\boldsymbol{\eta}^{\top} T(\mathbf{x})-A(\boldsymbol{\eta})\right) \\ &=\frac{1}{Z(\boldsymbol{\eta})} h(\mathbf{x}) \exp \left(\boldsymbol{\eta}^{\top} T(\mathbf{x})\right) \end{aligned}
$$

+ Moment estimation 
+ Generalized linear models 
+ Parameter estimation of GLIMs 

## Ensemble Methods

**Model Averaging**

Bagging: Average Trees and produces **smoother** decision boundaries

Random Forests: Cleverer averaging of tress

Boosting: Best

Boosting > Random Forests > Bagging > Single Tree 

### boosting

+ Average many trees, each grow to re-weighted versions of the training data

+ Final Classifier is weighted average of classifiers
  $$
  C(x) = sign \left[\sum_{m=1}^M \alpha_mC_m(x) \right]
  $$
  

**AdaBoost**

<<<<<<< HEAD
AdaBoost 模型是弱分类器的线性组合：
$$
f(x) = \sum_{m=1}^M \alpha_m G_m(x)
$$
算法的特点是通过迭代每次学习一个基本分类器，每次迭代中，提高那些被前一轮分类器错误分类的数据的权值，而降低那些被正确分类的数据的权值。最后将基本分类器的线性组合作为强分类器，其中给分类误差率小的基本分类器以大的权值，给分类误差率大的基本分类器以小的权值。

AdaBoost 的每次迭代可以减小其在训练集上的分类误差率。

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
![](https://i.loli.net/2019/04/21/5cbbda48ed596.png)

**Stage-wise Additive Modeling**

Boosting builds an additive model
$$
f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
$$
Traditionally the parameters $f_m,\theta_m$ are fit **jointly**, with boosting, the parameters are fit in a **stage-wise** fashion.(This slows the process down, and over-fits less quickly.)

+ Ada-boost builds an **addictive logistic regression model**
  $$
  f(x)=\log \frac{\operatorname{Pr}(Y=1 | x)}{\operatorname{Pr}(Y=-1 | x)}=\sum_{m=1}^{M} \alpha_{m} G_{m}(x)
  $$
  by stage-wise fitting using the loss fucntion
  $$
  L(y, f(x))=\exp (-y f(x))
  $$

+ Given the current $f_{M-1}()x$, our solution for $(\beta_m,G_m)$ is
  $$
  \arg \min _{\beta, G} \sum_{i=1}^{N} \exp \left[-y_{i}\left(f_{m-1}\left(x_{i}\right)+\beta G(x)\right)\right]
  $$

## Deep Learning

**Importance of Activation Functions**

To introduce non-linearities into the network

ReLu 

+ Constant gradient results in faster learning
+ **Increasing the sparsity of the network**
+ Solving the *Vanishing gradient*

### Back Propagation

### CNN

+ Convolution
+ Subsampling
+ Fully connected

## Unsupervised Learning :Clustering 

### K-means

+ the opt. problem
  $$
  \begin{array}{c}{\min _{\left\{C_{k}\right\}_{k=1}^{K}} \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_{k}}\left\|\mathbf{x}-\boldsymbol{\mu}_{k}\right\|_{2}^{2}} \\ {\text { s.t : } \quad \boldsymbol{\mu}_{k}=\frac{1}{\left|C_{k}\right|} \sum_{\mathbf{x} \in C_{k}} \mathbf{x}}\end{array}
  $$

+ Theorem: K-means iteratively leads to a non-increasing of the objective, until **local minimum** is achieved 

### Mixture of Gaussians and EM algorithm 

Likelihood is
$$
p\left(\mathcal{D} | \mu, \sigma^{2}\right)=\prod_{n=1}^{N} p\left(x_{n} | \mu, \sigma^{2}\right)
$$
MLE estimate 
$$
(\mu_{ML},\sigma_{ML}^2) = {argmax}_{\mu,\sigma^2} log_p(D|\mu,\sigma^2)
$$

$$
\begin{aligned} \mu_{\mathrm{ML}}=& \frac{1}{N} \sum_{n=1}^{N} x_{n} \\ \sigma_{\mathrm{ML}}^{2}=& \frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2} \quad \text { sample variance } \end{aligned}
$$

Note: MLE for the variance of a Gaussian is biased 

+ d-dimensional multivariate Gaussian 
  $$
  p(\mathbf{x} | \mu, \Sigma)=\frac{1}{(2 \pi)^{d / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mu) \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
  $$

$$
p(\mathcal{D} | \mu, \Sigma)=\prod_{n=1}^{N} p\left(\mathbf{x}_{n} | \mu, \Sigma\right)
$$

$$
\left(\mu_{\mathrm{ML}}, \Sigma_{\mathrm{ML}}\right)=\underset{\mu, \Sigma}{\operatorname{argmax}} \log p(\mathcal{D} | \mu, \Sigma)
$$

$$
\begin{array}{l}{\mu_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n} \quad \text { sample mean }} \\ {\Sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)\left(x_{n}-\mu_{\mathrm{ML}}\right)^{\top} \quad \text { sample covariance }}\end{array}
$$

**Mixture of Gaussians**
$$
p(x) = \sum_{k=1}^K \pi_k N(x|\mu_k,\small \sum_k)
$$
Log-likelihood
$$
\log p(\mathcal{D} | \pi, \mu, \Sigma)=\sum_{n=1}^{N} \log \left(\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x} | \mu_{k}, \Sigma_{k}\right)\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{k}}=0
$$

$$
\sum_{n=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{j}, \Sigma_{j}\right)} \Sigma_{k}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)=0
$$

$$
\boldsymbol{\mu}_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right) \mathbf{x}_{n} \quad N_{k}=\sum_{n=1}^{N} \gamma\left(z_{n k}\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial \Sigma_{k}}=0
$$

$$
\Sigma_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}
$$

Optimal Conditions
$$
L=\mathcal{L}(\boldsymbol{\mu}, \Sigma)+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right)
$$

$$
\frac{\partial L}{\partial \pi_{k}}=0
$$

$$
\begin{array}{l}{\sum_{n=1}^{N} \frac{\mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{j}, \Sigma_{j}\right)}+\lambda=0} \\ {\pi_{k}=\frac{N_{k}}{N}}\end{array}
$$

Summary

+ The set of couple conditions
  $$
  \begin{aligned} \boldsymbol{\mu}_{k} &=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right) \mathbf{x}_{n} \\ \Sigma_{k} &=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \\ \pi_{k} &=\frac{N_{k}}{N} \end{aligned}
  $$

+ The key factor to get them coupled
  $$
  \gamma\left(z_{n k}\right)=\frac{\pi_{k} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{j}, \Sigma_{j}\right)}
  $$

### The EM Algorithm

+ E-step : estimate the responsibilities 
  $$
  \gamma\left(z_{n k}\right)=\frac{\pi_{k} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\mathbf{x}_{n} | \boldsymbol{\mu}_{j}, \Sigma_{j}\right)}
  $$

+ M-step : re-estimate the parameters 
  $$
  \begin{aligned} \boldsymbol{\mu}_{k} &=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right) \mathbf{x}_{n} \\ \Sigma_{k} &=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma\left(z_{n k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \\ \pi_{k} &=\frac{N_{k}}{N} \end{aligned}
  $$

### Theory

+ Jensen’s inequality 
  $$
  \log \mathbb{E}_{p(x)}[x] \geq \mathbb{E}_{p(x)}[\log x]
  $$

$$
\begin{aligned} \log p(\mathcal{D} | \Theta) &=\sum_{n=1}^{N} \log \left(\sum_{\mathbf{z}_{n}} q\left(\mathbf{z}_{n}\right) \frac{p\left(\mathbf{x}_{n}, \mathbf{z}_{n}\right)}{q\left(\mathbf{z}_{n}\right)}\right) \\ & \geq \sum_{n=1}^{N} \sum_{\mathbf{z}_{n}} q\left(\mathbf{z}_{n}\right) \log \left(\frac{p\left(\mathbf{x}_{n}, \mathbf{z}_{n}\right)}{q\left(\mathbf{z}_{n}\right)}\right) \end{aligned}
$$

+ We have a lower bound
  $$
  \log p(\mathcal{D} | \Theta) \geq \sum_{n=1}^{N} \sum_{\mathbf{z}_{n}} q\left(\mathbf{z}_{n}\right) \log \left(\frac{p\left(\mathbf{x}_{n}, \mathbf{z}_{n}\right)}{q\left(\mathbf{z}_{n}\right)}\right) \triangleq \mathcal{L}(\Theta, q(\mathbf{Z}))
  $$

+ What is the GAP?
  $$
  \log p(\mathcal{D} | \Theta)-\mathcal{L}(\Theta, q(\mathbf{Z}))=\operatorname{KL}(q(\mathbf{Z}) \| p(\mathbf{Z} | \mathcal{D})
  $$

![](https://i.loli.net/2019/04/21/5cbbe45457211.png)

Maximize the lower bound or minimize the gap :

+ E-step : Maximize over $q(Z)$
+ M-step: Maximize over $\theta$

The EM algorithm for GMM reduces to K-Means under certain conditions

<<<<<<< HEAD
EM算法在每次迭代后均提高观测数据的似然函数值，即
$$
P(Y|\theta^{(i+1)}) \geq P(Y|\theta^{(i)})
$$
在一般条件下 EM 算法是收敛的，但不能保证收敛到全局最优解。

=======
>>>>>>> f5bea4e7e85d92e6c6aa7fa601671bcdb16129fd
## Unsupervised Learning (II) Dimension Reduction 
### PCA: Principal Component Analysis 

#### Maximum Variance Formulation 

For 1-dimensional projection, we care about the direction so we assume
$$
\mu_1^T\mu_1 =1
$$
Projection is
$$
y_n = \mu_1^T x_n
$$
Mean and variance of projected data:
$$
\bar{y} = \mu_1 \bar{x},\quad \bar{x} = \frac{1}{N}\sum_n x_n
$$

$$
var(y) =\frac{1}{N}\sum_n(\mu_1^Tx_n - \mu_1^T\bar{x})^2 = \mu_1^TS\mu_1
$$

And the sample covariance
$$
S = \frac{1}{N}\sum_n (x_n - \bar{x})(x_n - \bar{x})^T
$$
Then the constrained optimization problem
$$
max_{\mu_1} \quad var(y)=\frac{1}{N}\sum_n (\mu_1^Tx_n-\mu_1^T\bar{x})^2 = \mu_1^TS\mu_1
$$
where $\mu_1^T\mu_1 =1$

Solve with Lagrangian methods, we get
$$
S\mu_1 = \lambda_1\mu_1
$$
The Lagrange multiplier is the eigenvalue
$$
\mu_1^TS\mu_1 = \lambda_1
$$
The eigenvector corresponds to largest eigenvalue is $1^{st}$ PC

Additional components
$$
\max _{\mu_{2}} \quad \operatorname{var}(y)=\frac{1}{N} \sum_{n}\left(\mu_{2}^{\top} \mathbf{x}_{n}-\mu_{2}^{\top} \overline{\mathbf{x}}\right)^{2}=\mu_{2}^{\top} S \mu_{2}
$$
where $
\mu_{2}^{\top} \mu_{2}=1 \quad \text { and } \quad \mu_{1}^{\top} \mu_{2}=0
$

We get
$$
S \mu_{2}-\lambda_{2} \mu_{2}-\gamma \mu_{1}=0
$$
Multiplying $\mu_1^T$, we get
$$
\gamma=\mu_{1}^{\top} S \mu_{2}=\lambda_{1} \mu_{1}^{\top} \mu_{2}=0
$$
Thus
$$
S \mu_{2}=\lambda_{2} \mu_{2} \quad \mu_{2}^{\top} S \mu_{2}=\lambda_{2}
$$

#### Minimum Error Formulation

**Comments**

+ Non-parametric 
+ Does not deal properly with missing data 
+ Outlying data observations can unduly affect the analysis 

### Probabilistic PCA

### Locally linear embedding (LLE) 

A nonlinear dimension reduction technique to preserve neighborhood structure 

**Points**

+ Motivation for dimension reduction
+ Derivation of PCA
+ LLE
+ Feature selection

