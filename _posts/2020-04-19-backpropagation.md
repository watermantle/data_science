---
title: "Let's write a keras-style MLP with numpy (1)"
date: 2020-04-19
tags: [backpropagation]
header:
  image: "images/backpropagation/cover.png"
excerpt: "Machine Learning, Backpropagation, Data Science"
---
## Preface

**keras** is a building-blocks-like package for deep learning and well-known for its flexibility. In This
article, we will apply backpropagation and write a keras like MLP (multilayer perceptron) with numpy.

### 1. What is perceptron, how it works?

Before we get into MLP, we obviously need to understand what is perceptron.
In essence,

formula block:
\[p(\theta) = \mathbf{\prod}_{i,c}p(\mathbf{\theta}^i(c))\]
