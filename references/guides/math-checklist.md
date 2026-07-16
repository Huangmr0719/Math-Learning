# 数学推导审查清单

## 记号

- 所有符号是否在使用前定义？
- 标量、向量、矩阵、张量和随机变量是否明确区分？
- 是否说明 batch、feature、time 和 sample 维度？

## 代数与微积分

- 等式变形的符号、常数和转置是否正确？
- 被省略的常数是否确实与待优化参数无关？
- 梯度是否可以用有限差分验证？
- 交换求导、积分和期望时是否声明适用条件？

## 概率

- 分布是否归一化，支持集条件是否满足？
- 期望是对哪个分布计算的？
- KL 方向是否写清，零概率边界是否正确处理？
- 是否区分 conditional、marginal、joint 和单个样本？

## 动力系统

- ODE/SDE 的时间方向、初值和终值是否明确？
- drift、diffusion、score 和 velocity 的约定是否一致？
- change of variables 是否包含绝对值和 Jacobian determinant？
- flow 是否检查 source 和 target distribution？

## 实现

- 数学公式与 tensor shape、广播和 reduction 轴是否一致？
- 随机实验是否固定种子？
- 解析值是否与数值值进行小规模对比？
- 数值实验是否被准确描述为证据，而不是数学证明？

