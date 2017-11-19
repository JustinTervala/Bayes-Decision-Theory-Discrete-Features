slidenumbers: true
<br>
# 2.9 Bayes Decision Theory
## Discrete Features
-
-
### Yuta Akizuki

---

## Recap of 2.2 - Bayesian Formula

The posterior probability computed from $$p(\boldsymbol{x}|\omega_j)$$ is

$$
P(w_{j}|\boldsymbol{x}) = \cfrac{p(\boldsymbol{x}|\omega_{j})P(w_{j})}{p(\boldsymbol{x})}
$$

- The categories: $${\omega_1, ..., \omega_c}$$
- The feature vector of $$\ \boldsymbol{x}=(x_1,...,x_d)^t$$
- $$x_i \in \boldsymbol{R}$$

---

## Recap of 2.3 - Zero-one Loss Function

The conditional risk is

$$
\begin{eqnarray*}
R(\alpha_{i} | \boldsymbol{x})&=&\displaystyle \sum_{j=1}^c \lambda(\alpha_{i}|\omega_{j})P(\omega_{j}|\boldsymbol{x})\\
&=&1 - P(\omega_{i}|\boldsymbol{x})
\end{eqnarray*}
$$

where the zero-one loss function is assigned,

$$
\lambda(\alpha_{i}|\omega_{j}) = \left\{
\begin{array}{ll}
0 & (i = j) \\
1 & (i \neq j)
\end{array}
\right.
$$

---

## Recap of 2.4 - Discriminant Functions

Let $$g_{i}(\boldsymbol{x}) = -R(\alpha_{i} | \boldsymbol{x})$$ as a discriminant function, the classifier is said to assign a feature vector $$\boldsymbol{x}$$ to class $$\omega_{i}$$ if

$$
g_{i}(\boldsymbol{x}) > g_{j}(\boldsymbol{x}) \quad for \: all \: j \neq i
$$

where

$$
g_{i}(\boldsymbol{x}) = P(\omega_{i}|\boldsymbol{x}) =  \cfrac{p(\boldsymbol{x}|\omega_{i})P(w_{i})}{\displaystyle \sum_{j=1}^c p(\boldsymbol{x}|\omega_{j})P(\omega_{j})}
$$

---

## Recap of 2.4 - Discriminant Functions


Since the evidence is independent on $$\omega_{i}$$ and natural logarithm is a monotonically increasing function, $$g_{i}(\boldsymbol{x})$$ can be written as

$$
g_{i}(\boldsymbol{x}) =\ln p(\boldsymbol{x}|\omega_{i}) + \ln P(\omega_i)
$$

For the two category case, the discriminant function is defined as

$$
\begin{eqnarray*}
g(\boldsymbol{x}) &\equiv& g_1(\boldsymbol{x}) - g_2(\boldsymbol{x})\\
&=&\ln \cfrac{p(\boldsymbol{x}|\omega_{1})}{p(\boldsymbol{x}|\omega_{2})} + \ln \cfrac{P(\omega_{1})}{P(\omega_{2})}
\end{eqnarray*}
$$

---

# 2.9 Bayes Decision Theory
## Discrete Features

---

## 2.9 Bayes Decision Theory - Discrete Features

When the components of $$\boldsymbol{x}$$ are discrete values, sums of discrete probability distribution becomes

$$
\sum_x P(\boldsymbol{x}|\omega_{j})
$$

instead of integrals of the probability density function for continuous features:

$$
\int p(\boldsymbol{x}|\omega_{j}) d \boldsymbol{x}
$$

---

## 2.9 Bayes Decision Theory - Discrete Features

The posterior probability is

$$
P(w_{j}|\boldsymbol{x}) = \cfrac{P(\boldsymbol{x}|\omega_{j})P(w_{j})}{P(\boldsymbol{x})}
$$

Where the evidence is

$$
P(\boldsymbol{x}) =\displaystyle \sum_{j=i}^c P(\boldsymbol{x}|\omega_{j})P(\omega_{j})
$$

---

## 2.9 Bayes Decision Theory - Discrete Features

To minimize the error rate is to maximum the posterior probability when the loss function$$\ \lambda(\alpha_{i}|\omega_{j})$$ is zero-one loss function.


$$
\begin{eqnarray*}
\alpha^{*}&=&\arg\min_{i} R(\alpha_{i} | \boldsymbol{x})\\
&=&\arg\min_{i} \displaystyle \sum_{j=i}^c \lambda(\alpha_{i}|\omega_{j})P(\omega_{j}|\boldsymbol{x})\\
&=&\arg\min_{i} 1 - P(\omega_{i}|\boldsymbol{x})\\
&=&\arg\max_{i} P(\omega_{i}|\boldsymbol{x})\\
\end{eqnarray*}
$$

---

## 2.9.1 Independent Binary Features

Consider the two-category problem:

- The categories: $$\omega_1, \omega_2$$
- The independent feature vector of $$\ \boldsymbol{x}=(x_1,...,x_d)^t$$
- $$x_i= 0\;, \; 1$$
- $$p_i=Pr[x_i=1|\omega_1]$$ (the probability of $$x_i=1$$ under $$\omega_1$$)
- $$q_i=Pr[x_i=1|\omega_2]$$ (the probability of $$x_i=1$$ under $$\omega_2$$)

---

## 2.9.1 Independent Binary Features

The class-conditional probabilities can be written as:

$$
P(\boldsymbol{x}|\omega_1)=\prod_{i=1}^d p_i^{x_i}(1-p_i)^{1-{x_i}} \quad P(\boldsymbol{x}|\omega_2)=\prod_{i=1}^d q_i^{x_i}(1-q_i)^{1-{x_i}}
$$

The likelihood ratio is given by

$$
\cfrac{P(\boldsymbol{x}|\omega_1)}{P(\boldsymbol{x}|\omega_2)}=\prod_{i=1}^d \left(\cfrac{p_i}{q_i}\right)^{x_i} \left(\cfrac{1-p_i}{1-q_i}\right)^{1-{x_i}}
$$

---

## 2.9.1 Independent Binary Features

The discriminant function is

$$
\begin{eqnarray*}
g(\boldsymbol{x})&=&ln \cfrac{p(\boldsymbol{x}|\omega_{1})}{p(\boldsymbol{x}|\omega_{2})} + \cfrac{P(\omega_{1})}{P(\omega_{2})}\\
\\
&=&\displaystyle \sum_{i=1}^d \left[x_i \ln \cfrac{p_i}{q_i} + (1-x_i) \ln \cfrac{1-p_i}{1-q_i} \right] + \ln \cfrac{P(\omega_1)}{P(\omega_2)}
\end{eqnarray*}
$$

---

## 2.9.1 Independent Binary Features

The function is linear in the $$x_i$$, and thus can be written

$$
g(\boldsymbol{x}) =\displaystyle \sum_{i=1}^d w_i x_i + w_0
$$

where

$$
w_i =\ln \cfrac{p_i(1-q_i)}{q_i(1-p_i)} \quad and \quad
w_0= \displaystyle \sum_{i=1}^d \ln \cfrac{1-p_i}{1-q_i} + \ln \cfrac{P(\omega_1)}{P(\omega_2)}
$$

---

## 2.9.1 Independent Binary Features

$$
w_i =\ln \cfrac{p_i}{q_i} \cfrac{1-q_i}{1-p_i} \; and \;
w_0= \displaystyle \sum_{i=1}^d \ln \cfrac{1-p_i}{1-q_i} + \ln \cfrac{P(\omega_1)}{P(\omega_2)}
$$

- If $$p_i > q_i$$, then $$\frac{p_i}{q_i} > 0, \frac{1-q_i}{1-p_i} > 0$$ $$\Rightarrow$$ $$w_i > 0$$
- For any fixed $$q_i < 1$$, $$\omega_i$$ gets larger as $$p_i$$ gets larger
- $$P(w_i)$$ give biases the decision in favor of $$\omega_i$$ in the threshold weight of $$w_0$$

---

## 2.9.1 Independent Binary Features

- The simple classifier is obtained because of the condition of feature independence.
- The inter-dependent features needs a more complicated classifier.
- The possible values for $$\boldsymbol{x}$$ appear in d-dimensional hypercube, and the decision surface defined by $$g(\boldsymbol{x}) = 0$$ is a hyperplane.

---

## Bayesian Decisions for 3D Binary Data

$$
P(\omega_1) = P(\omega_2) = 0.5, \; p_i = 0.8 \; and \; q_i = 0.5 \; for \; i = 1,2,3
$$

The decision surface[^1] is

$$
g(\boldsymbol{x})=\sum_{i=1}^3 1.3863 x_i -2.75 = 0
$$

$$
1.3863x_1+1.3863x_2+1.3863x_3-2.75=0
$$

[^1]: Visualized decision surface for example 3 on [GitHub](https://ytakzk.github.io/Bayes-Decision-Theory-Discrete-Features/).
