"""课程专业术语的统一中英对照。

教材正文遵循“中文先行、英文对照”的规则：术语第一次出现时给出通用
中文译名、论文中的英文写法和一句话解释；后续优先使用中文或已经声明的
缩写。没有稳定中文译名的术语会明确说明，而不是强行创造译名。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermSpec:
    chinese: str
    english: str
    explanation: str
    introduced_in: int


TERMS: dict[str, TermSpec] = {
    # 第 1–7 章：VAE 基础
    "autoencoder": TermSpec("自编码器", "autoencoder", "把输入压缩成表示，再尝试重建输入的模型。", 1),
    "encoder": TermSpec("编码器", "encoder", "把输入数据映射到较低维表示或分布参数的网络。", 1),
    "decoder": TermSpec("解码器", "decoder", "根据潜在表示恢复或生成数据的网络。", 1),
    "latent_representation": TermSpec("潜在表示", "latent representation", "模型在内部学到的压缩表示；也常简称 latent。", 1),
    "latent_space": TermSpec("潜空间", "latent space", "所有潜在表示所在的空间，也常译作“隐空间”。", 1),
    "batch": TermSpec("批次", "batch", "一次并行送入模型的一组样本。", 1),
    "tensor_shape": TermSpec("张量形状", "tensor shape", "张量每个轴的长度及其语义，例如批次轴和特征轴。", 1),
    "broadcasting": TermSpec("广播", "broadcasting", "按明确规则扩展长度为 1 的轴，以完成逐元素运算。", 1),
    "reduction": TermSpec("归约", "reduction", "通过求和或平均等操作消去某些张量轴。", 1),
    "latent_variable": TermSpec("潜变量", "latent variable", "不能直接观察、但用于解释观测数据的随机变量。", 2),
    "prior": TermSpec("先验分布", "prior", "看到当前观测之前，对潜变量的概率判断。", 2),
    "likelihood": TermSpec("似然", "likelihood", "固定潜在原因后，当前观测出现的可能性。", 2),
    "marginal_likelihood": TermSpec("边缘似然", "marginal likelihood", "把所有潜在原因的贡献汇总后得到的观测概率。", 2),
    "posterior": TermSpec("后验分布", "posterior", "看到观测之后，对潜变量更新得到的概率分布。", 3),
    "evidence": TermSpec("模型证据", "evidence", "贝叶斯公式的归一化分母；在这里就是边缘似然。", 3),
    "approximate_inference": TermSpec("近似推断", "approximate inference", "真实后验难以计算时，用可计算分布近似它。", 3),
    "expectation": TermSpec("期望", "expectation", "按概率加权的长期平均值。", 4),
    "kl_divergence": TermSpec("KL 散度", "Kullback–Leibler divergence", "衡量一个分布用另一个分布近似时的信息损失；方向不可交换。", 4),
    "monte_carlo": TermSpec("蒙特卡洛估计", "Monte Carlo estimation", "用随机样本平均近似难以直接计算的期望或积分。", 4),
    "elbo": TermSpec("证据下界", "evidence lower bound, ELBO", "对数边缘似然的可优化下界。", 5),
    "jensen": TermSpec("詹森不等式", "Jensen's inequality", "把凸函数或凹函数与期望联系起来的不等式。", 5),
    "gaussian": TermSpec("高斯分布", "Gaussian distribution", "由均值和方差描述的连续概率分布，也称正态分布。", 6),
    "reparameterization": TermSpec("重参数化技巧", "reparameterization trick", "把随机性移到固定噪声中，使分布参数仍可反向传播。", 6),
    "log_variance": TermSpec("对数方差", "log-variance, logvar", "方差的自然对数；代码中常用它保证还原出的方差为正。", 6),
    "binary_cross_entropy": TermSpec("二元交叉熵", "binary cross-entropy, BCE", "用于二值数据或伯努利输出的负对数似然损失。", 7),

    # 第 8–13 章：VAE 变体
    "beta_vae": TermSpec("β-VAE", "beta-VAE", "用系数 β 调整 KL 项权重的 VAE 变体。", 8),
    "disentanglement": TermSpec("解耦表示", "disentangled representation", "让不同潜在维度尽量对应相对独立的变化因素。", 8),
    "lagrangian": TermSpec("拉格朗日乘子法", "Lagrangian method", "把约束问题转写成带权目标的优化方法。", 8),
    "posterior_collapse": TermSpec("后验坍塌", "posterior collapse", "近似后验退化得接近先验，潜变量几乎不再携带输入信息。", 9),
    "kl_annealing": TermSpec("KL 权重退火", "KL annealing", "训练早期减小 KL 权重，再逐渐提高的策略。", 9),
    "free_bits": TermSpec("自由比特", "free bits", "给部分 KL 信息量留出不立即惩罚的额度；业内常保留英文。", 9),
    "iwae": TermSpec("重要性加权自编码器", "Importance-Weighted Autoencoder, IWAE", "使用多个重要性样本构造更紧变分下界的模型。", 10),
    "importance_sampling": TermSpec("重要性采样", "importance sampling", "从较易采样的分布取样，再用权重修正目标分布期望。", 10),
    "logsumexp": TermSpec("LogSumExp 数值技巧", "log-sum-exp trick", "先减最大值再计算指数和，以避免数值上溢。", 10),
    "cvae": TermSpec("条件变分自编码器", "Conditional VAE, CVAE", "把类别或其他条件同时加入推断与生成过程的 VAE。", 11),
    "one_hot": TermSpec("独热编码", "one-hot encoding", "用只有一个位置为 1 的向量表示离散类别。", 11),
    "vq_vae": TermSpec("向量量化变分自编码器", "Vector-Quantized VAE, VQ-VAE", "用有限码本构造离散潜在表示的自编码器。", 12),
    "codebook": TermSpec("码本", "codebook", "一组可学习的离散表示向量。", 12),
    "vector_quantization": TermSpec("向量量化", "vector quantization", "把连续向量替换为码本中最近的向量。", 12),
    "straight_through": TermSpec("直通估计器", "straight-through estimator, ST", "前向使用不可导离散操作，反向用近似梯度传递信号。", 12),
    "normalizing_flow": TermSpec("归一化流", "normalizing flow", "通过一系列可逆变换构造灵活概率分布的模型。", 13),
    "change_of_variables": TermSpec("变量变换公式", "change-of-variables formula", "用局部体积缩放修正变换后的概率密度。", 13),
    "jacobian": TermSpec("雅可比矩阵", "Jacobian matrix", "由多元函数所有一阶偏导数组成的矩阵。", 13),
    "determinant": TermSpec("行列式", "determinant", "在变量变换中描述局部有向体积的缩放倍数。", 13),

    # 第 14–20 章：扩散模型与 DDIM
    "markov_chain": TermSpec("马尔可夫链", "Markov chain", "下一状态的条件分布只需读取当前状态的随机过程。", 14),
    "transition_kernel": TermSpec("转移核", "transition kernel", "规定从当前状态到下一状态条件分布的规则。", 14),
    "forward_process": TermSpec("前向过程", "forward process", "从数据逐步加入噪声的过程。", 14),
    "ddpm": TermSpec("去噪扩散概率模型", "Denoising Diffusion Probabilistic Models, DDPM", "学习逆转逐步加噪过程的生成模型。", 15),
    "noise_schedule": TermSpec("噪声调度", "noise schedule", "规定每个时间步加入多少噪声的一组系数。", 15),
    "reverse_process": TermSpec("反向过程", "reverse process", "从高噪声状态逐步恢复数据的生成过程。", 16),
    "gaussian_conditioning": TermSpec("高斯条件分布运算", "Gaussian conditioning", "由联合高斯分布得到给定观测后的条件高斯分布。", 16),
    "score": TermSpec("得分函数", "score function", "概率密度对输入的对数梯度，即 ∇ₓ log p(x)；不是模型评分。", 17),
    "self_supervised": TermSpec("自监督学习", "self-supervised learning", "训练标签由数据本身的变换自动构造，而不依赖人工标注。", 17),
    "time_embedding": TermSpec("时间嵌入", "time embedding", "把离散或连续时间编码成网络可使用的向量。", 18),
    "sampler": TermSpec("采样器", "sampler", "根据模型预测和时间规则逐步生成样本的算法。", 18),
    "ddim": TermSpec("去噪扩散隐式模型", "Denoising Diffusion Implicit Models, DDIM", "与 DDPM 共享训练目标、但可采用非马尔可夫采样路径的模型。", 19),
    "deterministic_sampling": TermSpec("确定性采样", "deterministic sampling", "固定初始噪声和模型后，不再额外注入随机噪声的采样方式。", 19),
    "inversion": TermSpec("反演", "inversion", "把真实样本近似映射回模型的噪声轨迹。", 20),
    "trajectory": TermSpec("轨迹", "trajectory", "状态随离散步或连续时间形成的完整路径。", 20),

    # 第 21–26 章：扩散模型变体与连续时间
    "prediction_parameterization": TermSpec("预测参数化", "prediction parameterization", "选择让网络预测噪声、干净数据、得分或速度等不同坐标。", 21),
    "snr": TermSpec("信噪比", "signal-to-noise ratio, SNR", "信号强度与噪声强度的比值。", 21),
    "cfg": TermSpec("无分类器引导", "classifier-free guidance, CFG", "组合有条件与无条件预测来增强条件控制的方法。", 22),
    "conditional_prediction": TermSpec("有条件预测", "conditional prediction", "网络在读取条件信息时给出的预测。", 22),
    "unconditional_prediction": TermSpec("无条件预测", "unconditional prediction", "网络不读取具体条件时给出的预测。", 22),
    "latent_diffusion": TermSpec("潜空间扩散", "latent diffusion", "在压缩后的潜空间而非原始像素空间中运行扩散。", 23),
    "latent_scaling": TermSpec("潜变量缩放", "latent scaling", "用固定系数调整潜变量数值尺度，并在解码前恢复。", 23),
    "unet": TermSpec("U 形网络", "U-Net", "通过下采样、上采样和跳跃连接结合多尺度特征的网络。", 24),
    "dit": TermSpec("扩散 Transformer", "Diffusion Transformer, DiT", "用 Transformer 处理图像块序列的扩散去噪网络。", 24),
    "patch": TermSpec("图像块", "patch", "把图像切分得到的局部小块。", 24),
    "token": TermSpec("序列单元", "token", "Transformer 处理的基本序列元素；视觉模型中常直接保留英文。", 24),
    "self_attention": TermSpec("自注意力", "self-attention", "让每个序列位置根据内容聚合其他位置的信息。", 24),
    "numerical_solver": TermSpec("数值求解器", "numerical solver", "用有限步近似微分方程连续轨迹的算法。", 25),
    "euler_method": TermSpec("欧拉法", "Euler method", "用当前点的斜率向前走一步的一阶数值方法。", 25),
    "heun_method": TermSpec("休恩法", "Heun's method", "先预测终点斜率，再用首尾斜率平均进行校正的二阶方法。", 25),
    "nfe": TermSpec("函数评估次数", "number of function evaluations, NFE", "采样过程中调用神经网络或速度函数的总次数。", 25),
    "ode": TermSpec("常微分方程", "ordinary differential equation, ODE", "用确定性导数描述状态随时间变化的方程。", 26),
    "sde": TermSpec("随机微分方程", "stochastic differential equation, SDE", "同时包含确定性漂移和随机扩散的连续时间方程。", 26),
    "brownian_motion": TermSpec("布朗运动", "Brownian motion", "增量独立且服从方差等于时间长度的高斯分布的随机过程。", 26),
    "drift": TermSpec("漂移项", "drift", "SDE 中决定平均运动方向的确定性部分。", 26),
    "diffusion_coefficient": TermSpec("扩散系数", "diffusion coefficient", "SDE 中缩放布朗随机增量强度的系数。", 26),
    "probability_flow_ode": TermSpec("概率流常微分方程", "probability flow ODE", "用确定性 ODE 复现相关 SDE 边缘分布演化的方程。", 26),

    # 第 27–30 章：连续流与流匹配
    "cnf": TermSpec("连续归一化流", "Continuous Normalizing Flow, CNF", "用 ODE 的流映射连续运输概率分布的模型。", 27),
    "vector_field": TermSpec("向量场", "vector field", "为空间中每个位置和时间指定运动方向与速度的函数。", 27),
    "divergence": TermSpec("散度", "divergence", "描述向量场在局部使体积膨胀或收缩的标量。", 27),
    "continuity_equation": TermSpec("连续性方程", "continuity equation", "表达概率质量在速度场中守恒的偏微分方程。", 27),
    "flow_map": TermSpec("流映射", "flow map", "把初始状态沿 ODE 轨迹映射到某个时间状态的函数。", 27),
    "flow_matching": TermSpec("流匹配", "Flow Matching", "直接回归概率路径速度场的训练方法。", 28),
    "conditional_expectation": TermSpec("条件期望", "conditional expectation", "已知部分信息后，对剩余随机性的概率加权平均。", 28),
    "marginal_velocity": TermSpec("边缘速度场", "marginal velocity field", "只依赖当前状态和时间、并推动边缘分布演化的速度场。", 28),
    "coupling": TermSpec("耦合", "coupling", "保持两个指定边缘分布不变的联合配对分布。", 29),
    "optimal_transport": TermSpec("最优传输", "Optimal Transport, OT", "在所有合法耦合中寻找期望运输成本最小者。", 29),
    "transport_cost": TermSpec("运输成本", "transport cost", "衡量把一个样本位置移动到另一个位置所需代价的函数。", 29),
    "rectified_flow": TermSpec("Rectified Flow", "Rectified Flow", "通过重新配对或再训练使路径更直的方法；中文译名尚未完全统一，业内通常保留英文。", 29),
}


def terms_for_chapter(number: int) -> tuple[TermSpec, ...]:
    """Return terms whose first complete explanation belongs to a chapter."""

    return tuple(term for term in TERMS.values() if term.introduced_in == number)
