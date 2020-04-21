---
title: "Let's write a keras-style MLP with numpy (1)"
date: 2020-04-19
tags: [backpropagation]
header:
  overlay_image: "images/backpropagation/cover.png"
excerpt: "Machine Learning, Backpropagation, Data Science"
mathjax: true
---
## Preface

**keras** is a building-blocks-like package for deep learning and well-known for its flexibility. In This
article, we will apply backpropagation and write a keras like MLP (multilayer perceptron) with numpy.

### 1. What is MLP and how to train a MLP


formula block:
```
\[p(\theta) = \mathbf{\prod}_{i,c}p(\mathbf{\theta}^i(c))\]
```
