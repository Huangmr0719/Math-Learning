"""课程章节的单一事实来源。

首页、章节导航和骨架 notebook 都从这里读取标题、问题、前置知识和桥梁，
避免同一条学习路径在多个文件中逐渐分叉。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChapterSpec:
    number: int
    part: str
    title: str
    filename: str
    question: str
    prior: tuple[str, ...]
    new_math: tuple[str, ...]
    bridge: str
    status: str = "骨架"


PARTS = {
    "part01_vae": "第一部分：从表示学习到变分自编码器（VAE）",
    "part02_vae_variants": "第二部分：VAE 变体",
    "part03_diffusion_ddpm": "第三部分：扩散模型与去噪扩散概率模型（DDPM）",
    "part04_ddim": "第四部分：DDIM",
    "part05_diffusion_variants": "第五部分：扩散模型变体",
    "part06_flow_matching": "第六部分：流匹配（Flow Matching）",
}


def _chapter(
    number: int,
    part: str,
    filename: str,
    title: str,
    question: str,
    prior: tuple[str, ...],
    new_math: tuple[str, ...],
    bridge: str,
    *,
    status: str = "骨架",
) -> ChapterSpec:
    return ChapterSpec(
        number=number,
        part=part,
        title=title,
        filename=filename,
        question=question,
        prior=prior,
        new_math=new_math,
        bridge=bridge,
        status=status,
    )


CHAPTERS = {
    1: _chapter(1, "part01_vae", "01_autoencoder_and_latent.py", "自编码器（Autoencoder）与潜在表示", "压缩和重构是否等于学会生成？", ("向量可以表示一条数据", "函数把输入映射成输出"), ("向量维度", "欧氏距离", "均方误差"), "重构成功并不保证潜空间可随机采样。", status="精写"),
    2: _chapter(2, "part01_vae", "02_latent_distribution.py", "从潜在点到潜在分布", "如何让潜空间连续、可采样并具有概率意义？", ("自编码器把样本编码成点", "潜在点之间可能存在空洞"), ("随机变量", "概率密度", "条件概率", "边缘概率"), "边缘似然包含难以直接计算的积分。", status="精写"),
    3: _chapter(3, "part01_vae", "03_bayesian_posterior.py", "贝叶斯后验与近似推断", "给定观测 x，怎样反推最可能产生它的 z？", ("联合概率描述变量共同发生", "条件概率描述已知条件后的可能性"), ("贝叶斯公式", "先验分布", "似然", "后验分布"), "真实后验往往不可计算，需要近似分布。", status="精写"),
    4: _chapter(4, "part01_vae", "04_kl_divergence.py", "KL 散度与变分思想", "怎样衡量近似分布与目标分布的差异？", ("概率分布的总和为 1", "对数把乘法变成加法"), ("期望", "信息量", "KL 散度", "蒙特卡洛估计"), "KL 散度可以把后验逼近问题转成优化目标。", status="精写"),
    5: _chapter(5, "part01_vae", "05_elbo.py", "证据下界（ELBO）的完整推导", "无法直接最大化 log p(x) 时，怎样得到可训练目标？", ("贝叶斯公式", "KL 散度非负", "期望是加权平均"), ("詹森不等式", "变分下界", "证据差距"), "ELBO 中仍有随机采样，下一步要让采样可微。", status="精写"),
    6: _chapter(6, "part01_vae", "06_gaussian_and_reparameterization.py", "高斯后验与重参数化", "怎样让随机采样参与反向传播？", ("ELBO 包含 q(z|x) 下的期望", "编码器要输出一个分布"), ("高斯分布", "对数方差", "仿射变换", "链式法则"), "VAE 所需的概率组件和可训练估计器已经齐全。", status="精写"),
    7: _chapter(7, "part01_vae", "07_minimal_vae.py", "最小 VAE 完整闭环", "如何把概率图、ELBO、神经网络和实验现象对应起来？", ("近似后验", "ELBO", "重参数化", "Gaussian KL"), ("batch reduction", "二元交叉熵"), "VAE 组件闭环后，先回到 AEVB 原论文核对问题、式号、算法和证据，再进入结构变体。", status="精写"),
    8: _chapter(8, "part02_vae_variants", "08_beta_vae.py", "β-VAE 与表示约束", "重构质量与潜在表示的规整程度如何权衡？", ("VAE 损失由重构项和 KL 项组成",), ("加权多目标优化", "拉格朗日乘子法直觉"), "KL 太强时，模型可能完全不使用潜变量。", status="精写"),
    9: _chapter(9, "part02_vae_variants", "09_posterior_collapse.py", "后验坍塌与训练策略", "为什么 q(z|x) 会退化成 p(z)？", ("KL 越小表示后验越接近先验",), ("互信息直觉", "KL 权重退火", "自由比特"), "标准 ELBO 的估计质量仍可改进。", status="精写"),
    10: _chapter(10, "part02_vae_variants", "10_iwae.py", "IWAE 与更紧的变分下界", "增加 posterior 样本能否获得更紧的下界？", ("ELBO 是 log evidence 的下界",), ("importance sampling", "log-sum-exp", "估计方差"), "除了改进估计，还可以改变条件信息。", status="精写"),
    11: _chapter(11, "part02_vae_variants", "11_conditional_vae.py", "条件 VAE", "怎样控制 VAE 生成的内容？", ("VAE 同时包含编码器和解码器条件分布",), ("条件联合分布", "条件独立"), "连续高斯潜变量并不是唯一选择。", status="精写"),
    12: _chapter(12, "part02_vae_variants", "12_vq_vae.py", "VQ-VAE 与离散潜变量", "怎样学习离散的潜在码本？", ("编码器产生潜在表示",), ("最近邻", "向量量化", "直通估计器"), "离散或连续潜变量都可以进一步增强后验。", status="精写"),
    13: _chapter(13, "part02_vae_variants", "13_flexible_posterior.py", "更灵活的后验与变量变换", "对角高斯后验表达能力不足怎么办？", ("概率密度会随空间变换改变",), ("雅可比矩阵", "行列式", "变量变换公式"), "可以不显式编码后验，而从噪声逐步恢复数据。", status="精写"),
    14: _chapter(14, "part03_diffusion_ddpm", "14_iterative_denoising.py", "从一次生成到逐步去噪", "为什么把困难的生成任务拆成许多小步骤？", ("Gaussian 噪声可方便采样",), ("Markov chain", "transition kernel"), "需要得到任意时间步的直接加噪公式。", status="精写"),
    15: _chapter(15, "part03_diffusion_ddpm", "15_ddpm_forward.py", "DDPM 前向过程", "怎样从 x0 一步采样任意 xt？", ("Gaussian 的线性变换仍是 Gaussian",), ("均值传播", "方差传播", "累计乘积"), "已知 xt 后，还需推导 xt-1 的条件分布。", status="精写"),
    16: _chapter(16, "part03_diffusion_ddpm", "16_ddpm_posterior_and_reverse.py", "DDPM 后验与反向过程", "真实反向条件分布是什么形式？", ("Bayes rule", "Gaussian 密度"), ("Gaussian conditioning", "完成平方"), "生成时没有 x0，需要神经网络预测它或等价量。", status="精写"),
    17: _chapter(17, "part03_diffusion_ddpm", "17_diffusion_elbo_and_noise_prediction.py", "从扩散模型 ELBO 到噪声预测", "为什么 DDPM 最终训练成预测噪声的 MSE？", ("ELBO", "高斯 KL", "马尔可夫链"), ("链式联合分布", "得分函数"), "目标函数明确后，可以构造最小 DDPM。", status="精写"),
    18: _chapter(18, "part03_diffusion_ddpm", "18_minimal_ddpm.py", "最小 DDPM 实现", "如何把 forward、训练目标和 reverse sampling 写成代码？", ("噪声预测目标", "时间条件"), ("sinusoidal embedding",), "同一个网络可以使用不同的采样轨迹。", status="精写"),
    19: _chapter(19, "part04_ddim", "19_ddim_sampling.py", "DDIM 与确定性生成轨迹", "同一个训练网络为什么可以使用不同采样过程？", ("DDPM 噪声预测网络",), ("条件方差", "确定性映射"), "确定性轨迹还能反向映射真实图像。", status="精写"),
    20: _chapter(20, "part04_ddim", "20_ddim_inversion.py", "DDIM 反演与图像编辑", "怎样把真实图像映射回噪声轨迹？", ("DDIM 确定性更新",), ("数值逆过程", "重构误差"), "采样器差异可以放入更统一的参数化和求解器视角。", status="精写"),
    21: _chapter(21, "part05_diffusion_variants", "21_prediction_parameterizations.py", "预测参数化与噪声路径", "模型应该预测 epsilon、x0、score 还是 v？", ("xt 是 x0 与噪声的线性组合",), ("SNR", "线性参数变换"), "条件生成需要把外部信息加入预测场。", status="精写"),
    22: _chapter(22, "part05_diffusion_variants", "22_classifier_free_guidance.py", "条件生成与无分类器引导（CFG）", "怎样控制生成内容而不单独训练分类器？", ("条件概率", "得分函数或噪声预测"), ("有条件得分", "线性外推"), "高分辨率生成还需要降低扩散空间的计算成本。", status="精写"),
    23: _chapter(23, "part05_diffusion_variants", "23_latent_diffusion.py", "潜空间扩散（Latent Diffusion）", "为什么不直接在像素空间扩散？", ("VAE/VQ-VAE 可压缩图像", "DDPM 可在任意连续空间训练"), ("压缩率", "感知失真"), "去噪网络本身也可以从 U-Net 更换为 Transformer。", status="精写"),
    24: _chapter(24, "part05_diffusion_variants", "24_unet_to_dit.py", "从 U 形网络（U-Net）到扩散 Transformer（DiT）", "去噪网络必须是卷积 U-Net 吗？", ("时间条件网络预测噪声或速度",), ("图像块嵌入", "自注意力直觉"), "固定模型后，采样速度取决于数值求解方法。", status="精写"),
    25: _chapter(25, "part05_diffusion_variants", "25_numerical_solvers.py", "快速采样与数值求解器", "如何用更少模型调用完成采样？", ("轨迹由局部方向逐步推进",), ("Euler method", "Heun method", "局部截断误差"), "离散 diffusion 可以写成连续时间动力系统。", status="精写"),
    26: _chapter(26, "part05_diffusion_variants", "26_score_sde_and_probability_flow_ode.py", "得分函数、随机微分方程与概率流 ODE", "离散扩散如何统一到连续时间？", ("得分函数", "欧拉法"), ("常微分方程", "随机微分方程", "布朗运动", "Fokker–Planck 直觉"), "既然 ODE 能运输分布，下一步可以直接学习速度场。", status="精写"),
    27: _chapter(27, "part06_flow_matching", "27_continuous_normalizing_flow.py", "连续归一化流（CNF）", "如何通过 ODE 把简单分布运输到数据分布？", ("ODE 描述状态随时间变化",), ("向量场", "流映射", "连续性方程"), "最大似然 CNF 追踪密度较昂贵，能否直接监督速度？", status="精写"),
    28: _chapter(28, "part06_flow_matching", "28_flow_matching.py", "流匹配（Flow Matching）目标", "不知道边缘速度场时，怎样训练它？", ("概率路径", "向量场", "期望"), ("条件期望", "边缘化"), "路径选择会影响轨迹弯曲程度和求解难度。", status="精写"),
    29: _chapter(29, "part06_flow_matching", "29_optimal_transport_and_rectified_flow.py", "最优传输与 Rectified Flow", "怎样选择更直、更容易求解的运输路径？", ("流匹配回归速度场",), ("耦合", "运输成本", "线性插值"), "最后需要统一比较 VAE、扩散模型与流匹配。", status="精写"),
    30: _chapter(30, "part06_flow_matching", "30_unified_view.py", "生成模型的统一视角", "这些模型在哪些地方相同，又在哪些地方真正不同？", ("VAE", "DDPM", "DDIM", "Flow Matching"), ("统一坐标系",), "回到研究问题：根据任务选择正确的分布建模方式。", status="精写"),
}
