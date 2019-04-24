## Statistical ML Test

#### 1 [20 pts] True or False

1. You are training a classification model with logistic regression. Which of the following statements are true? Check all that apply. (**TFFF**)
   - The decision boundary of multi-class logistic regression is piecewise linear.
   - Adding new features to the model helps prevent overfitting on the training set.
   - Introducing regularization to the model always results in equal or better performance on the training set. （实际上在 training set 是会变差的）
   - Introducing regularization to the model always results in equal or better performance on examples not in the training set.
2. Please answer the following True/False statements for ensemble methods: (**TTFF**)
   - Bagging, random forests, boosting are all ensemble methods.
   - Suppose in every AdaBoost iteration we get a weak classifier $$C_{m}$$ with margin $$\gamma>0$$, we can achieve ZERO error training dataset after finite iterations.
     (Recall a weak classifier $$C_m$$ with margin $$\gamma$$ satisfies $$\sum_{i=1}^mw^{(i)}1\{y^{(i)} \ne C_m({\rm x}^{(i)})\} \leq \frac{1}{2}-\gamma$$)  
   - AdaBoost will not lead to *overfitting*, which means the testing error will decrease monotonously during the training time.
   - The Random Forests method fits many large trees to bootstrap-resampled versions of the training data and classify by majority vote.
3. Please answer the following True/False statements for clustering and EM: (**TFFT**) 最后一个答案是F
   - K-means is a special case of Gaussian Mixture Model (GMM) under some conditions.
   - In general, K-means converges to global minimum of its objective.
   - In EM for a mixture of Gaussians, where $$\rm Z​$$ denotes the component indicators, the gap between the lower bound and the true data log likelihood is $${\rm KL}(q({\rm Z})||p({\rm Z}))​$$.
   - In the E step of EM, the lower bound is raised to match the true data log likelihood. So the M step is optimizing the data log likelihood over model parameters.
4. Please answer the following True/False statements for the kernel trick: (**TFTT**) 
   - The kernel trick can be applied to PCA since the covariance matrix is formed by inner-products.
   - The Gram matrix of a kernel is positive definite.
   - The feature mapping $$\phi:{\rm x}\rightarrow\phi({\rm x})$$ defined by the RBF kernel $$k({\rm x}, {\rm x}')=e^{-\frac{||{\rm x}-{\rm x}'||^2}{2\sigma^2}}$$ maps $${\rm x}$$ into an infinite-dimensional space.
   - There exists a dual form of the *Perceptron* algorithm where the kernel trick can be applied.



#### 2 [40 pts] Short Questions

1. In SVM, why do we need to relax the 0-1 loss into the hinge loss? Besides the hinge loss, list two smooth approximations of the 0-1 loss.

   0-1 Loss 是非凸的 **对于连续和可导性有要求，在实际计算中便于使用；Quadratic loss , Huber loss**

2. Under which assumptions will Gaussian Naive Bayes be equivalent to Logistic Regression?

   **GNB with class-independent variances representationally equivalent to LR**

3. What are the support vectors in the Soft SVM?

   ![](https://i.loli.net/2019/04/22/5cbdc99d59654.png)

   **Margin support vectors**  $y_iw^Tx_i =1$

   **Non-Margin support vectors**  $\xi_i > 0$

   对应于 $\alpha^{\star}>0​$的点$x_i​$称为支持向量，其中如图$x_I​$到间隔边界的距离为 $\frac{\xi_i}{||w||}​$

   如果 $\alpha^{\star}<C$，则$\xi_i=0$，支持向量恰好落在间隔边界上；若$\alpha^{\star}=C, 0<\xi_i<1$,对应分类正确，支持向量在间隔边界与分离超平面之间；若$\alpha^{\star}=C,\xi_i=1$，则支持向量在分离超平面上；若$\alpha^{\star}=C,\xi_i>1$，则支持向量位于分离超平面误分一侧。

4. Explain why non-linear activation functions are important in neural networks. Taking the sigmoid activation as an example, explain what *gradient vanishing* is and why.

   **The purpose of activation functions is to introduce non-linearities into the network** 

   What is :在神经网络中，当前面隐藏层的学习速率低于后面隐藏层的学习速率，即随着隐藏层数目的增加，分类准确率反而下降了。这种现象叫做消失的梯度问题。

   For sigmoid function, sigmoid函数中心和两侧的梯度差别太大，如果权重初始变化的太大，激活值几乎都在sigmoid两侧，但两侧的梯度几乎接近0，进行几层的传播就没有梯度了。

5. Consider a centered data matrix $$\rm X = [ x_1, ..., x_n]^{\top}$$, where $${\rm {x_i}}\in\mathbb{R}^d, \sum_{i=1}^n \rm x_i = 0.$$ When $$d \gg n$$, it is extremely time-consuming to work with  $$\rm X^{\top}X$$. Under this condition, propose a method that finds the direction $$\rm f \in\mathbb{R}^d$$ that maximizes the variance of the projections of $$\rm x_{1:n}$$ onto $$\rm f$$.

   **To be solved** 

   

6. For an exponential family distribution:

   $$
   p({\rm x}|\eta)=h({\rm x}){\rm exp}(\eta^{\top}T({\rm x})-A(\eta))
   $$

   Derive both the 1st and 2nd order derivatives of $$A(\eta)$$ in its general form and show how they relate to the mean and (co)-variance of $$T(\rm x)$$.

   **指数族分布**

   For Exponential family， we have 
   $$
   \begin{aligned} p(\mathbf{x} | \boldsymbol{\eta}) &=h(\mathbf{x}) \exp \left(\boldsymbol{\eta}^{\top} T(\mathbf{x})-A(\boldsymbol{\eta})\right) \\ &=\frac{1}{Z(\boldsymbol{\eta})} h(\mathbf{x}) \exp \left(\boldsymbol{\eta}^{\top} T(\mathbf{x})\right) \end{aligned}
   $$
   And 
   $$
   A(\eta) = \log Z(\eta) = \log \int h(x)\exp(\eta^TT(x))dx
   $$
   有
   $$
   \frac{\partial A}{\partial \eta^T} = \frac{\partial}{\partial \eta^T} \left\{\log \int \exp(\eta^T T(x)h(x)dx) \right\}
   $$
   Then
   $$
   \frac{\partial A}{\partial \eta^T} = \frac{\int T(x)  \exp(\eta^T T(x)h(x))dx}{\int \exp(\eta^T T(x)h(x))dx}
   $$

   $$
   = \int T(x)\exp(\eta^T T(x)h(x)-A(\eta)h(x))dx
   $$

   $$
   = E[T(x)]
   $$

   For a second derivative:
   $$
   \begin{aligned} \frac{\partial^{2} A}{\partial \eta \partial \eta^{T}} &=\int T(x)\left(T(x)-\frac{\partial}{\partial \eta^{T}} A(\eta)\right]^{T} \exp \left\{\eta^{T} T(x)-A(\eta)\right\} h(x) (d x) \\ &=\int T(x)(T(x)-\mathbb{E}[T(X)])^{T} \exp \left\{\eta^{T} T(x)-A(\eta)\right\} h(x)(d x) \\ &=\mathbb{E}\left[T(X) T(X)^{T}\right]-\mathbb{E}[T(X)] \mathbb{E}[T(X)] )^{T} \\ &=\operatorname{Var}[T(X)] \end{aligned}
   $$

7. Consider using a logistic regression model $$\sigma({\rm w}^{\top}{\rm x})$$ where $$\sigma$$ is the sigmoid function, and let a training set $$\{({\rm x}_i, y_i)\}_{i=1}^m$$ is given. The maximum likelihood estimate of $$\rm w$$ is given by

   $$
   {\rm w_{MLE}}=\mathop{\arg\max}_{\rm w}\prod_{i=1}^mp(y_i|{\rm x}_i, \rm w).
   $$

   If we wanted to regularize logistic regression, then we might put a Bayessian prior on the parameters. Suppose we chose the prior $${\rm w}\sim \mathcal N(0, r^2{\rm I})$$ where $$r>0$$ and $$\rm I$$ is the identity matrix. Then the maximum a posterior estimate is 

   $$
   {\rm w_{MAP}}=\mathop{\arg\max}_{\rm w}p({\rm w})\prod_{i=1}^mp(y_i|{\rm x}_i, \rm w).
   $$

   Please fill in the blank: $$||\rm w_{MAP}||_2$$ _ _ _ _ (≥, ≤ or =) $$||\rm w_{MLE}||_2$$ and explain why.

   $\leq​$ 

   **Reasons** 

   最大后验估计类似于正则化的效果：

   假设参数服从标准正态： $p(W)=\prod_{j} N\left(W_{j} | 0, \tau^{2}\right)$ 

   最大后验概率估计就是：
   $$
   \begin{array}{c}{\operatorname{argmax}_{w} \log \left(\prod_{i=1}^{n} N\left(y_{i} | W^{T} x_{i}, \sigma^{2}\right) P(W)\right)} \\ {=\sum_{i=1}^{n} \log N\left(y_{i} | W^{T} x_{i}, \sigma^{2}\right)+\sum_{j} \log N\left(W_{j} | 0, \tau^{2}\right)}\end{array}
   $$
   后半部分就是正则化项

8. In back-propagation, what is the signal propagated from an upper layer to the lower layer?

   **误差反向传播** 

   $\frac{\partial E}{\partial W_k}$

#### 3 [20 pts] EM

We derive the EM algorithm for the mixture of **TWO** components. Suppose that we have two known density functions $$f(x)​$$ and $$g(x)​$$. We generate sample $$x_i​$$ as follows:

- Draw an indicator $$z_i \sim {\rm Bernoulli}(\theta)\  ({\rm i.e.}\ P(z_i =1)=\theta)​$$. It tells which component of the mixture $$z_i​$$ comes from.
- If $$z_i =1$$, draw $$x_i\sim f(z_i)$$, else ($$z_i=0$$) draw $$x_i\sim g(z_i)$$.

Now suppose that we observe a dataset $$\mathcal{D}=(x_1, ..., x_n)​$$. Answer the following questions:

1. Derive a lower bound $$L(\mathcal{D}; \theta, q)​$$ for $$\log p(\mathcal{D}|\theta)​$$ by introducing distributions $$q(z_i)​$$ for each $$z_i​$$.
2. E step: show the optimal $$q(z_i)$$ for maximizing  $$L(\mathcal{D}; \theta, q)$$  given $$\theta$$.
3. M step: show the updating rule for $$\theta^{(t)}$$ at the $$t$$-th iteration is

$$
\theta^{(t+1)}=\frac{1}{n}\sum_{i=1}^n\frac{\theta^{(t)}f(x_i)}{\theta^{(t)}f(x_i)+(1-\theta^{(t)})g(x_i)}.
$$

![WeChat Image_20190423101620](C:\Users\feiyuxiao\Desktop\WeChat Image_20190423101620.jpg) 



