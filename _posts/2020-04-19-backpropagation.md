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

### 1. What is perceptron, how it works?

Before we get into MLP, we obviously need to understand what is perceptron.
For simplicity, let's consider a binary classification problem with a decision function **$$ \phi(z) $$** that
has two possible outputs 1 and -1, representing two classes respectively, where
**z** is a linear combination with input **$$ x_i $$** and corresponding weights **$$ w_i $$**. Hence,
**$$ Z = w_1x_1 + w_2x_2 + ... + w_mx_m $$**:
$$ mathbf{W = \[[W_1], [...], [W_m]]}, mathbf{X = \[[X_1], [...], [X_m]]} $$

formula block:
$$ p(\theta) = \mathbf{\prod}_{i,c}p(\mathbf{\theta}^i(c)) $$

$$ sum_(i=1)^n i^3=((n(n+1))/2)^2 $$
