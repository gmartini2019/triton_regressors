# Introduction

Classical regression models are usually treated as solved problems.
On CPU, that is mostly true.
On GPU, they are not.

This repository exists to explore what happens when we:
- take classical regression seriously on modern accelerators
- separate training algorithms from inference kernels
- measure optimization behavior instead of hiding it

The goal is not convenience.
The goal is understanding.

Building this repository has helped me dive back into classical machine learning, hardware optimization and the mathematics behind both of these disciplines, I hope the same will be for you.