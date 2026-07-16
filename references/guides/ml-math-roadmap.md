# 生成模型数学路线图

这是一张导航图，不是需要提前学完的先修课程。数学概念应在对应模型章节中即时学习。

## VAE

- 向量、欧氏距离、均方误差
- 随机变量、概率密度、条件概率与 Bayes rule
- 期望、Monte Carlo、KL divergence
- Jensen inequality 与 ELBO
- Gaussian、方差、链式法则与重参数化

## VAE 变体

- 加权目标和 Lagrange multiplier
- aggregated posterior 与 mutual information
- importance sampling 与 log-sum-exp
- 条件独立、离散 latent 与 surrogate gradient
- Jacobian、determinant 与 change of variables

## Diffusion、DDPM 与 DDIM

- Markov chain 与 Gaussian transition
- 均值、方差和累计乘积
- Gaussian conditioning 与完成平方
- score、SNR 和预测参数化
- 离散采样路径与数值误差

## 连续时间与 Flow Matching

- 导数、ODE、Euler 和 Heun method
- Brownian motion、SDE 与 Fokker–Planck equation
- vector field、divergence 与 continuity equation
- conditional expectation 与 MSE 正交分解
- coupling、transport cost 与 Optimal Transport

学习时从当前章节链接进入所需概念，不建议脱离模型问题一次性学习整张列表。

