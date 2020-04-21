---
title: "Let's write a keras-style MLP with numpy (1)"
date: 2020-04-19
tags: [backpropagation]
header:
  image: "images/backpropagation/cover.png"
excerpt: "Machine Learning, Backpropagation, Data Science"
mathjax: "true"
---
## Preface

**keras** is a building-blocks-like package for deep learning and well-known for its flexibility. In This
article, we will apply backpropagation and write a keras like MLP (multilayer perceptron) with numpy.

#### 1. What is perceptron, how it works?

Before we get into MLP, we obviously need to understand what is perceptron.
For simplicity, let's consider a binary classification problem with a decision function $$ \mathbf{\phi(z)} $$ that
has two possible outputs 1 and -1, representing two classes respectively, where
$$ \mathbf{z} $$ is a linear combination with input $$ \mathbf{x_i} $$ and corresponding weights $$ \mathbf{w_i} $$. Hence,

$$ \mathbf{Z = w_1x_1 + w_2x_2 + ... + w_mx_m} = \sum_{i=1}^m w_ix_i $$ or:

$$ \mathbf{W = \left[\begin{matrix} W_1 \\ ... \\ W_m \end{matrix}\right]},
 \mathbf{X = \left[\begin{matrix} X_1 \\ ... \\ X_m \end{matrix}\right]} $$

 if we add a **bias term** of **b** to $$ \mathbf{z} $$, or here for uniformity, we set $$x_0 = 1$$ and
 $$w_0 = b$$, we modify our formula to:

$$ \mathbf{Z = w_0x_0 + w_1x_1 + w_2x_2 + ... + w_mx_m = \sum_{i=0}^m w_ix_i = W^TX} $$
and our decision function:
$$\mathbf{\phi(z)} =
\begin{aligned}
\begin{matrix} 1 if z >= 0 \\ -1 otherwise
\end{matrix}
\end{aligned}$$

$ [[w_0], [w_1]] $

Next we will do following steps to train a perceptron:
1. Initialize the weighs to small random numbers, we here can use stand normal distribution to do so.
2. For each training example, $$\mathbf{x^(i)}$$:
