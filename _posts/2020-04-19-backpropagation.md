---
title: "Let's write a keras-style MLP with numpy (1)"
date: 2020-04-19
tags: [backpropagation]
header:
  overlay_image: "images/Pittsburgh.jpg"
excerpt: "Machine Learning, Backpropagation, Data Science"
mathjax: "true"
---
## Preface

**keras** is a building-blocks-like package for deep learning and well-known for its flexibility. In This
article, we will apply backpropagation and write a keras like MLP (multilayer perceptron) with numpy.

#### 1. What is perceptron, how it works?

Before we get into MLP, we obviously need to understand what is perceptron.
For simplicity, let's consider a binary classification problem with a decision function $\phi(z)$ that
has two possible outputs 1 and -1, representing class 1 and class 2 respectively, where
$bb"z"$ is a linear combination with input $bb"x_i"$ and corresponding weights $bb"w_i"$. Hence,

**$Z = w_1x_1 + w_2x_2 + ... + w_mx_m = sum_(i=1)^m w_ix_i$** or:

**$W =  [[W_1], [...], [W_m]],
 X = [[X_1], [...], [X_m]]$**

 if we add a **bias** term, **b** to **z**, or here for uniformity, we set $x_0 = 1$ and
 $w_0 = b$, we hence modify our formula as:

**$Z = w_0x_0 + w_1x_1 + w_2x_2 + ... + w_mx_m = sum_(i=0)^m w_ix_i = W^TX$**

and our decision function:

**$\phi(z) = {(1,ifz>=0,,,),(-1,otherwise,,,):}$**


Next, we will do following steps to train a perceptron:
1. Initialize the weighs to small random numbers, we here can use stand normal distribution to do so.
2. For each training example, **$x^(i)$**:
    1. Compute the output values **$hat y$**, the one from decision function **$\phi(z)$**
    2. Update the weights [^1]

The process could be explained as follows:

![perceptron pic](/data_science/images/backpropagation/perceptron.png =600x370)



[^1]: We here will apply an approach called "Gradient Decent" to update weights. Since MLP
 and perceptron share the same method, we will explain it later.
