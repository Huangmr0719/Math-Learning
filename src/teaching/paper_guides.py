"""生成模型论文谱系与部分末经典论文导读元数据。

正式章节负责首次教学；论文坐标负责把章节概念定位回原始资料；部分末导读
训练整篇阅读、公式核验和研究判断。三者分工不同，避免把参考文献堆成书目。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperSource:
    """One primary paper or clearly labelled secondary reading."""

    key: str
    title: str
    chinese_title: str
    authors: str
    venue_year: str
    url: str
    role: str
    reading_targets: tuple[str, ...]
    chapters: tuple[int, ...]
    local_pdf: str = ""
    primary: bool = True


@dataclass(frozen=True)
class PaperGuideSpec:
    """A non-numbered paper seminar inserted after one formal part."""

    key: str
    part: str
    after_chapter: int
    order: int
    filename: str
    title: str
    paper_keys: tuple[str, ...]
    question: str


PAPERS = {
    "hinton_autoencoder": PaperSource(
        key="hinton_autoencoder",
        title="Reducing the Dimensionality of Data with Neural Networks",
        chinese_title="用神经网络降低数据维度",
        authors="Geoffrey E. Hinton 与 Ruslan R. Salakhutdinov",
        venue_year="Science, 2006",
        url="https://www.science.org/doi/10.1126/science.1127647",
        role="历史起点：深层自编码器如何把高维数据压进低维代码。",
        reading_targets=("图 1 的编码—解码结构", "重构目标与降维实验", "它为何尚未定义可采样的潜变量分布"),
        chapters=(1,),
    ),
    "denoising_autoencoder": PaperSource(
        key="denoising_autoencoder",
        title="Extracting and Composing Robust Features with Denoising Autoencoders",
        chinese_title="用去噪自编码器提取并组合稳健特征",
        authors="Pascal Vincent 等",
        venue_year="ICML, 2008",
        url="https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf",
        role="思想对照：损坏输入、恢复干净目标，说明重构约束可以塑造表示。",
        reading_targets=("损坏分布 q(x_tilde|x)", "去噪重构目标", "与 VAE 随机潜变量的本质区别"),
        chapters=(1,),
    ),
    "contractive_autoencoder": PaperSource(
        key="contractive_autoencoder",
        title="Contractive Auto-Encoders: Explicit Invariance During Feature Extraction",
        chinese_title="收缩自编码器：在特征提取中显式学习不变性",
        authors="Salah Rifai 等",
        venue_year="ICML, 2011",
        url="https://icml.cc/2011/papers/455_icmlpaper.pdf",
        role="思想对照：惩罚编码器 Jacobian，使表示对局部扰动不敏感。",
        reading_targets=("式 (1) 的 Jacobian 范数与式 (7) 的 CAE 目标", "图 1 的局部收缩曲线", "确定性正则化为何仍不等于概率生成模型"),
        chapters=(1,),
    ),
    "aevb": PaperSource(
        key="aevb",
        title="Auto-Encoding Variational Bayes",
        chinese_title="自编码式变分贝叶斯",
        authors="Diederik P. Kingma 与 Max Welling",
        venue_year="ICLR 2014 / arXiv:1312.6114v10",
        url="https://arxiv.org/abs/1312.6114",
        role="VAE 主干：摊销变分推断、SGVB 估计器与重参数化。",
        reading_targets=("式 (1)–(3) 的 ELBO", "式 (4)、(10) 的重参数化", "算法 1 与图 1"),
        chapters=(2, 3, 4, 5, 6, 7),
        local_pdf="references/papers/vae/auto-encoding-variational-bayes.pdf",
    ),
    "vae_tutorial": PaperSource(
        key="vae_tutorial",
        title="Tutorial on Variational Autoencoders",
        chinese_title="变分自编码器教程",
        authors="Carl Doersch",
        venue_year="arXiv:1606.05908, 2016",
        url="https://arxiv.org/abs/1606.05908",
        role="辅助教程：以视觉任务和图模型补足 AEVB 原文省略的教学台阶。",
        reading_targets=("第 2 节的概率模型", "第 2.2 节的损失", "第 3 节的条件 VAE"),
        chapters=(2, 3, 5, 6, 7, 11),
        local_pdf="references/papers/vae/tutorial-on-variational-autoencoders.pdf",
        primary=False,
    ),
    "beta_vae": PaperSource(
        key="beta_vae",
        title="beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework",
        chinese_title="β‑VAE：用受约束的变分框架学习基本视觉概念",
        authors="Irina Higgins 等",
        venue_year="ICLR, 2017",
        url="https://openreview.net/forum?id=Sy2fzU9gl",
        role="在 ELBO 的 KL 项前加入 β，研究表示容量与解耦的折中。",
        reading_targets=("式 (4) 的目标", "图 3–7 的潜变量遍历", "原论文指标的适用边界"),
        chapters=(8,),
    ),
    "beta_capacity": PaperSource(
        key="beta_capacity",
        title="Understanding Disentangling in beta-VAE",
        chinese_title="理解 β‑VAE 中的解耦",
        authors="Christopher P. Burgess 等",
        venue_year="arXiv:1804.03599, 2018",
        url="https://arxiv.org/abs/1804.03599",
        role="以率失真视角解释 β‑VAE，并提出逐渐增加目标容量 C。",
        reading_targets=("式 (8) 的 |KL-C| 目标", "容量调度图", "重构—解耦并非单调关系"),
        chapters=(8,),
    ),
    "bowman_text_vae": PaperSource(
        key="bowman_text_vae",
        title="Generating Sentences from a Continuous Space",
        chinese_title="从连续空间生成句子",
        authors="Samuel R. Bowman 等",
        venue_year="CoNLL, 2016 / arXiv:1511.06349",
        url="https://arxiv.org/abs/1511.06349",
        role="经典失败现场：强自回归解码器可能忽略潜变量，并引出 KL 退火。",
        reading_targets=("第 3.1 节的训练困难", "KL cost annealing", "语言模型负结果应怎样解读"),
        chapters=(9,),
    ),
    "lagging_inference": PaperSource(
        key="lagging_inference",
        title="Lagging Inference Networks and Posterior Collapse in Variational Autoencoders",
        chinese_title="滞后的推断网络与 VAE 后验坍塌",
        authors="Junxian He 等",
        venue_year="ICLR, 2019",
        url="https://openreview.net/forum?id=rylDfnCqF7",
        role="把坍塌解释为训练动态问题之一，并用 aggressive inference updates 诊断。",
        reading_targets=("图 1 的 inference lag", "算法 1", "机制解释不是唯一原因"),
        chapters=(9,),
    ),
    "iwae": PaperSource(
        key="iwae",
        title="Importance Weighted Autoencoders",
        chinese_title="重要性加权自编码器",
        authors="Yuri Burda、Roger Grosse 与 Ruslan Salakhutdinov",
        venue_year="ICLR, 2016 / arXiv:1509.00519",
        url="https://arxiv.org/abs/1509.00519",
        role="用 K 个重要性样本构造更紧的证据下界。",
        reading_targets=("式 (8)–(9) 的多样本下界", "Theorem 1 的单调趋紧条件", "K 增大不等于所有梯度都更好"),
        chapters=(10,),
    ),
    "cvae": PaperSource(
        key="cvae",
        title="Learning Structured Output Representation using Deep Conditional Generative Models",
        chinese_title="用深度条件生成模型学习结构化输出表示",
        authors="Kihyuk Sohn、Honglak Lee 与 Xinchen Yan",
        venue_year="NeurIPS, 2015",
        url="https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html",
        role="把 VAE 改写为条件似然问题，区分条件先验与识别网络。",
        reading_targets=("图 1(b)(c) 的两条路径", "式 (3)–(5) 的条件 ELBO", "训练时 q(z|x,y) 与测试时 p(z|x) 的差异"),
        chapters=(11,),
    ),
    "vq_vae": PaperSource(
        key="vq_vae",
        title="Neural Discrete Representation Learning",
        chinese_title="神经离散表示学习",
        authors="Aaron van den Oord、Oriol Vinyals 与 Koray Kavukcuoglu",
        venue_year="NeurIPS, 2017",
        url="https://papers.nips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html",
        role="以向量量化和直通估计器学习离散潜表示。",
        reading_targets=("式 (1) 的最近邻", "式 (3) 的三项损失", "图 1 的训练/先验两阶段"),
        chapters=(12,),
    ),
    "vq_vae_2": PaperSource(
        key="vq_vae_2",
        title="Generating Diverse High-Fidelity Images with VQ-VAE-2",
        chinese_title="用 VQ‑VAE‑2 生成多样、高保真图像",
        authors="Ali Razavi、Aaron van den Oord 与 Oriol Vinyals",
        venue_year="NeurIPS, 2019 / arXiv:1906.00446",
        url="https://arxiv.org/abs/1906.00446",
        role="拓展阅读：层次码本与自回归先验如何把离散表示扩展到大图像。",
        reading_targets=("图 1 的两层 latent", "图 2 的两阶段训练", "质量提升来自哪些新增部件"),
        chapters=(12,),
    ),
    "normalizing_flows": PaperSource(
        key="normalizing_flows",
        title="Variational Inference with Normalizing Flows",
        chinese_title="使用正规化流的变分推断",
        authors="Danilo Jimenez Rezende 与 Shakir Mohamed",
        venue_year="ICML, 2015 / arXiv:1505.05770",
        url="https://arxiv.org/abs/1505.05770",
        role="用可逆变换和 Jacobian 行列式构造更灵活的近似后验。",
        reading_targets=("式 (2)–(3) 的变量替换", "式 (5) 的 flow ELBO", "可逆性与 log-det 计算成本"),
        chapters=(13,),
    ),
    "td_vae": PaperSource(
        key="td_vae",
        title="Temporal Difference Variational Auto-Encoder",
        chinese_title="时序差分变分自编码器",
        authors="Karol Gregor 等",
        venue_year="ICLR, 2019 / arXiv:1806.03107",
        url="https://arxiv.org/abs/1806.03107",
        role="拓展阅读：把潜变量模型、belief state 与跨时间跳跃预测结合。",
        reading_targets=("belief state", "jumpy prediction", "阅读前先掌握第 14 章 Markov chain"),
        chapters=(),
    ),
    "diffusion_thermodynamics": PaperSource(
        key="diffusion_thermodynamics",
        title="Deep Unsupervised Learning using Nonequilibrium Thermodynamics",
        chinese_title="用非平衡热力学进行深度无监督学习",
        authors="Jascha Sohl-Dickstein、Eric A. Weiss、Niru Maheswaranathan 与 Surya Ganguli",
        venue_year="ICML, 2015 / arXiv:1503.03585",
        url="https://arxiv.org/abs/1503.03585",
        role="扩散概率模型的历史起点：固定逐步破坏过程，再学习有限时间的反向转移。",
        reading_targets=("第 2 节的 forward/reverse chains", "图 1 的 Swiss-roll 破坏与恢复", "变分下界与逐步训练目标"),
        chapters=(14, 15, 16, 17),
    ),
    "score_matching": PaperSource(
        key="score_matching",
        title="Estimation of Non-Normalized Statistical Models by Score Matching",
        chinese_title="用得分匹配估计未归一化统计模型",
        authors="Aapo Hyvärinen",
        venue_year="Journal of Machine Learning Research, 2005",
        url="https://www.jmlr.org/papers/v6/hyvarinen05a.html",
        role="得分匹配的基础来源：绕开未知归一化常数，直接拟合对数密度关于数据的梯度。",
        reading_targets=("第 2 节的 score 定义", "Theorem 1 的分部积分条件", "它为何不是分类模型的评分"),
        chapters=(17,),
    ),
    "denoising_score_matching": PaperSource(
        key="denoising_score_matching",
        title="A Connection Between Score Matching and Denoising Autoencoders",
        chinese_title="得分匹配与去噪自编码器之间的联系",
        authors="Pascal Vincent",
        venue_year="Neural Computation, 2011",
        url="https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and",
        role="把显式得分匹配改写为去噪目标，为噪声预测与得分估计之间的联系提供基础。",
        reading_targets=("式 (4)–(6) 的显式与去噪得分匹配", "加噪条件分布的 score", "等价结论所需的正则条件"),
        chapters=(17,),
    ),
    "ncsn": PaperSource(
        key="ncsn",
        title="Generative Modeling by Estimating Gradients of the Data Distribution",
        chinese_title="通过估计数据分布梯度进行生成建模",
        authors="Yang Song 与 Stefano Ermon",
        venue_year="NeurIPS, 2019 / arXiv:1907.05600",
        url="https://arxiv.org/abs/1907.05600",
        role="得分生成路线的关键节点：在多个噪声尺度学习得分，并用退火 Langevin 动力学生成。",
        reading_targets=("式 (1) 的 score", "式 (5) 的多尺度目标", "算法 1 的 annealed Langevin dynamics"),
        chapters=(17,),
    ),
    "ddpm": PaperSource(
        key="ddpm",
        title="Denoising Diffusion Probabilistic Models",
        chinese_title="去噪扩散概率模型",
        authors="Jonathan Ho、Ajay Jain 与 Pieter Abbeel",
        venue_year="NeurIPS, 2020 / arXiv:2006.11239",
        url="https://arxiv.org/abs/2006.11239",
        role="DDPM 主干：高斯前向闭式、反向均值参数化、变分目标与噪声预测训练。",
        reading_targets=("式 (1)–(7) 的两条 Markov chain", "式 (10)–(14) 的噪声参数化", "算法 1–2 与图 2"),
        chapters=(14, 15, 16, 17, 18),
    ),
    "improved_ddpm": PaperSource(
        key="improved_ddpm",
        title="Improved Denoising Diffusion Probabilistic Models",
        chinese_title="改进的去噪扩散概率模型",
        authors="Alexander Q. Nichol 与 Prafulla Dhariwal",
        venue_year="ICML, 2021 / PMLR 139",
        url="https://proceedings.mlr.press/v139/nichol21a.html",
        role="DDPM 的重要工程与似然改进：学习反向方差、混合目标、采样提速与规模规律。",
        reading_targets=("第 2.2 节的 learned variance", "式 (16) 的 hybrid objective", "图 8 的采样步数对照"),
        chapters=(18,),
    ),
    "ddim": PaperSource(
        key="ddim",
        title="Denoising Diffusion Implicit Models",
        chinese_title="去噪扩散隐式模型",
        authors="Jiaming Song、Chenlin Meng 与 Stefano Ermon",
        venue_year="ICLR, 2021 / arXiv:2010.02502v4",
        url="https://arxiv.org/abs/2010.02502",
        role="DDIM 主干：构造与 DDPM 共享训练目标的非马尔可夫过程，并得到可确定化、可跨步的采样器。",
        reading_targets=("图 1 的 Markov/non-Markov 对照", "式 (12) 的通用更新", "图 4–5 的步数、时间与一致性"),
        chapters=(19, 20),
    ),
    "null_text_inversion": PaperSource(
        key="null_text_inversion",
        title="Null-text Inversion for Editing Real Images using Guided Diffusion Models",
        chinese_title="用于真实图像编辑的空文本反演",
        authors="Ron Mokady、Amir Hertz、Kfir Aberman、Yael Pritch 与 Daniel Cohen-Or",
        venue_year="CVPR, 2023 / arXiv:2211.09794",
        url="https://arxiv.org/abs/2211.09794",
        role="DDIM inversion 的代表性后续：分析大 CFG 下的重构误差，并优化无条件文本嵌入。",
        reading_targets=("图 2 的 inversion/editing pipeline", "第 3.2 节的 null-text optimization", "不要把后续编辑方法倒写为 DDIM 原论文贡献"),
        chapters=(20,),
    ),
    "progressive_distillation": PaperSource(
        key="progressive_distillation",
        title="Progressive Distillation for Fast Sampling of Diffusion Models",
        chinese_title="用于扩散模型快速采样的渐进蒸馏",
        authors="Tim Salimans 与 Jonathan Ho",
        venue_year="ICLR, 2022 / arXiv:2202.00512",
        url="https://arxiv.org/abs/2202.00512",
        role="预测参数化与快速采样坐标：讨论 velocity 参数化并逐轮把采样步数减半。",
        reading_targets=("第 2.2 节的 velocity parameterization", "算法 2 的 progressive distillation", "参数化改善条件数不等于无条件优于所有目标"),
        chapters=(21,),
    ),
    "edm": PaperSource(
        key="edm",
        title="Elucidating the Design Space of Diffusion-Based Generative Models",
        chinese_title="阐明扩散生成模型的设计空间",
        authors="Tero Karras、Miika Aittala、Timo Aila 与 Samuli Laine",
        venue_year="NeurIPS, 2022 / arXiv:2206.00364",
        url="https://arxiv.org/abs/2206.00364",
        role="统一设计坐标：分离噪声参数化、预条件、训练权重、采样轨迹与数值求解器。",
        reading_targets=("表 1 的设计维度", "式 (7)–(8) 的预条件", "算法 1–2 的 deterministic/stochastic sampling"),
        chapters=(21, 25),
    ),
    "classifier_guidance": PaperSource(
        key="classifier_guidance",
        title="Diffusion Models Beat GANs on Image Synthesis",
        chinese_title="扩散模型在图像合成上超越生成对抗网络",
        authors="Prafulla Dhariwal 与 Alexander Q. Nichol",
        venue_year="NeurIPS, 2021 / arXiv:2105.05233",
        url="https://arxiv.org/abs/2105.05233",
        role="分类器引导的来源：用噪声分类器的对数概率梯度修正反向 score。",
        reading_targets=("式 (4) 的 classifier gradient", "图 6 的 precision/recall 权衡", "额外分类器必须适应 noisy inputs"),
        chapters=(22,),
    ),
    "classifier_free_guidance": PaperSource(
        key="classifier_free_guidance",
        title="Classifier-Free Diffusion Guidance",
        chinese_title="无分类器扩散引导",
        authors="Jonathan Ho 与 Tim Salimans",
        venue_year="arXiv:2207.12598v1, 2022",
        url="https://arxiv.org/abs/2207.12598",
        role="CFG 主干：联合训练有条件与无条件预测，并用两者差向量控制质量—多样性折中。",
        reading_targets=("算法 1–2", "图 2 的三高斯引导", "图 4–5 的 FID/IS trade-off"),
        chapters=(22,),
    ),
    "latent_diffusion": PaperSource(
        key="latent_diffusion",
        title="High-Resolution Image Synthesis with Latent Diffusion Models",
        chinese_title="使用潜空间扩散模型进行高分辨率图像合成",
        authors="Robin Rombach、Andreas Blattmann、Dominik Lorenz、Patrick Esser 与 Björn Ommer",
        venue_year="CVPR, 2022 / arXiv:2112.10752",
        url="https://arxiv.org/abs/2112.10752",
        role="潜空间扩散主干：先用自编码器压缩，再在感知相关的潜空间运行扩散，并以 cross-attention 注入条件。",
        reading_targets=("图 2 的感知/语义压缩", "图 3 的 LDM 与 cross-attention", "式 (1)–(3) 的训练目标"),
        chapters=(23,),
    ),
    "unet": PaperSource(
        key="unet",
        title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
        chinese_title="U 形网络：用于生物医学图像分割的卷积网络",
        authors="Olaf Ronneberger、Philipp Fischer 与 Thomas Brox",
        venue_year="MICCAI, 2015 / arXiv:1505.04597",
        url="https://arxiv.org/abs/1505.04597",
        role="U-Net 架构的原始来源；扩散模型借用了多尺度下采样、上采样与 skip connection，但任务和模块已经改变。",
        reading_targets=("图 1 的 contracting/expanding path", "skip connection 的空间对齐", "不要把原始分割 U-Net 等同于 DDPM U-Net"),
        chapters=(24,),
    ),
    "dit": PaperSource(
        key="dit",
        title="Scalable Diffusion Models with Transformers",
        chinese_title="使用 Transformer 扩展扩散模型",
        authors="William Peebles 与 Saining Xie",
        venue_year="ICCV, 2023 / arXiv:2212.09748",
        url="https://arxiv.org/abs/2212.09748",
        role="DiT 主干：用处理潜空间 patch 的 Transformer 替代 U-Net，并系统研究计算量与样本质量的缩放关系。",
        reading_targets=("图 3 的 DiT block", "图 2 与图 6–9 的 scaling", "结论限定于论文的数据、训练预算和模型族"),
        chapters=(24,),
    ),
    "dpm_solver": PaperSource(
        key="dpm_solver",
        title="DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps",
        chinese_title="DPM-Solver：约十步采样扩散概率模型的快速常微分方程求解器",
        authors="Cheng Lu、Yuhao Zhou、Fan Bao、Jianfei Chen、Chongxuan Li 与 Jun Zhu",
        venue_year="NeurIPS, 2022 / arXiv:2206.00927",
        url="https://arxiv.org/abs/2206.00927",
        role="专用数值求解器坐标：利用扩散 ODE 的半线性结构构造高阶更新，并以 NFE 比较效率。",
        reading_targets=("第 3 节的 diffusion ODE", "算法 1–3", "固定 NFE 而非只比较步数"),
        chapters=(25,),
    ),
    "score_sde": PaperSource(
        key="score_sde",
        title="Score-Based Generative Modeling through Stochastic Differential Equations",
        chinese_title="通过随机微分方程进行基于得分的生成建模",
        authors="Yang Song、Jascha Sohl-Dickstein、Diederik P. Kingma、Abhishek Kumar、Stefano Ermon 与 Ben Poole",
        venue_year="ICLR, 2021 / arXiv:2011.13456v2",
        url="https://arxiv.org/abs/2011.13456",
        role="连续时间统一主干：forward SDE、reverse-time SDE、predictor-corrector 与 probability flow ODE。",
        reading_targets=("图 2 的统一框架", "式 (5)–(7) 的 forward/reverse SDE", "式 (13) 的 probability flow ODE"),
        chapters=(26,),
    ),
    "neural_ode": PaperSource(
        key="neural_ode",
        title="Neural Ordinary Differential Equations",
        chinese_title="神经常微分方程",
        authors="Ricky T. Q. Chen、Yulia Rubanova、Jesse Bettencourt 与 David Duvenaud",
        venue_year="NeurIPS, 2018 / arXiv:1806.07366",
        url="https://arxiv.org/abs/1806.07366",
        role="CNF 的连续时间起点：用神经网络参数化状态导数，并给出瞬时变量替换公式。",
        reading_targets=("图 1 的离散层与连续向量场", "Theorem 1 的 instantaneous change of variables", "式 (8) 的 Jacobian trace 与适用条件"),
        chapters=(27,),
    ),
    "ffjord": PaperSource(
        key="ffjord",
        title="FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models",
        chinese_title="FFJORD：面向可扩展可逆生成模型的自由形式连续动力学",
        authors="Will Grathwohl、Ricky T. Q. Chen、Jesse Bettencourt、Ilya Sutskever 与 David Duvenaud",
        venue_year="ICLR, 2019 / arXiv:1810.01367",
        url="https://arxiv.org/abs/1810.01367",
        role="把 CNF 的精确 Jacobian trace 换成 Hutchinson 无偏估计，使高维密度训练更可扩展。",
        reading_targets=("图 1 的连续密度运输", "式 (7) 的 Hutchinson trace estimator", "Algorithm 1 的增广状态与随机估计边界"),
        chapters=(27,),
    ),
    "flow_matching": PaperSource(
        key="flow_matching",
        title="Flow Matching for Generative Modeling",
        chinese_title="用于生成建模的流匹配",
        authors="Yaron Lipman、Ricky T. Q. Chen、Heli Ben-Hamu、Maximilian Nickel 与 Matt Le",
        venue_year="ICLR, 2023 / arXiv:2210.02747",
        url="https://arxiv.org/abs/2210.02747",
        role="Flow Matching 主干：用可采样的条件概率路径和条件速度场，间接训练边缘 CNF 向量场。",
        reading_targets=("式 (5) 与式 (9) 的 FM/CFM 目标", "Theorem 1–2 的边缘化与梯度等价", "Figure 2–3 的 diffusion path 与 OT path"),
        chapters=(28, 29, 30),
    ),
    "rectified_flow": PaperSource(
        key="rectified_flow",
        title="Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow",
        chinese_title="让流更直、更快：用 Rectified Flow 生成与迁移数据",
        authors="Xingchao Liu、Chengyue Gong 与 Qiang Liu",
        venue_year="ICLR, 2023 / arXiv:2209.03003",
        url="https://arxiv.org/abs/2209.03003",
        role="Rectified Flow 主干：从任意 coupling 的线性插值监督学习因果 ODE，并用 reflow 改善诱导 coupling 与轨迹直度。",
        reading_targets=("式 (1) 的最小二乘目标", "Figure 2 的 crossing 与 rewiring", "Theorem 3.5 的凸运输成本不增边界"),
        chapters=(29, 30),
    ),
}


PAPER_GUIDES = {
    "vae_autoencoder_lineage": PaperGuideSpec(
        key="vae_autoencoder_lineage",
        part="part01_vae",
        after_chapter=7,
        order=1,
        filename="paper_guides/01_autoencoder_lineage.py",
        title="论文谱系导读：普通自编码器为什么还不是 VAE",
        paper_keys=("hinton_autoencoder", "denoising_autoencoder", "contractive_autoencoder", "aevb"),
        question="四篇论文都使用 encoder–decoder，为什么只有 AEVB 建立了可采样的概率生成模型？",
    ),
    "vae_aevb": PaperGuideSpec(
        key="vae_aevb",
        part="part01_vae",
        after_chapter=7,
        order=2,
        filename="paper_guides/02_auto_encoding_variational_bayes.py",
        title="经典论文深读：从 AEVB 原文重新发现 VAE",
        paper_keys=("aevb",),
        question="能否从论文问题、公式、图和算法重新构造出第 1–7 章的 VAE？",
    ),
    "vae_variants_landmarks": PaperGuideSpec(
        key="vae_variants_landmarks",
        part="part02_vae_variants",
        after_chapter=13,
        order=1,
        filename="paper_guides/01_vae_variants_landmarks.py",
        title="经典论文研讨：六个 VAE 缺口与六条改进路线",
        paper_keys=("beta_vae", "beta_capacity", "bowman_text_vae", "lagging_inference", "iwae", "cvae", "vq_vae", "vq_vae_2", "normalizing_flows"),
        question="每个变体究竟修改了目标、推断、条件、潜变量类型，还是后验族？",
    ),
    "diffusion_lineage": PaperGuideSpec(
        key="diffusion_lineage",
        part="part03_diffusion_ddpm",
        after_chapter=18,
        order=1,
        filename="paper_guides/01_diffusion_lineage.py",
        title="论文谱系导读：扩散链与得分场如何在 DDPM 会合",
        paper_keys=("diffusion_thermodynamics", "score_matching", "denoising_score_matching", "ncsn", "ddpm"),
        question="逐步逆转概率转移与学习数据分布的得分，为什么最终会导向同一种噪声预测网络？",
    ),
    "ddpm_original": PaperGuideSpec(
        key="ddpm_original",
        part="part03_diffusion_ddpm",
        after_chapter=18,
        order=2,
        filename="paper_guides/02_ddpm_original.py",
        title="经典论文深读：从 DDPM 原文重建训练与采样",
        paper_keys=("ddpm", "improved_ddpm"),
        question="能否从原文的式 (1)–(14) 与算法 1–2，严格重建前向加噪、训练目标和反向采样？",
    ),
    "ddim_original": PaperGuideSpec(
        key="ddim_original",
        part="part04_ddim",
        after_chapter=20,
        order=1,
        filename="paper_guides/01_ddim_original.py",
        title="经典论文深读：DDIM 为什么能换一条更短的采样路径",
        paper_keys=("ddim", "null_text_inversion"),
        question="保持训练目标和每时刻边缘分布时，为什么可以改用非马尔可夫、甚至确定性的生成过程？",
    ),
    "diffusion_variants_landmarks": PaperGuideSpec(
        key="diffusion_variants_landmarks",
        part="part05_diffusion_variants",
        after_chapter=26,
        order=1,
        filename="paper_guides/01_diffusion_variants_landmarks.py",
        title="经典论文研讨：扩散模型的六个可独立设计旋钮",
        paper_keys=("progressive_distillation", "edm", "classifier_guidance", "classifier_free_guidance", "latent_diffusion", "unet", "dit", "dpm_solver", "score_sde"),
        question="预测目标、条件引导、表示空间、网络骨干、求解器和时间形式分别改变了扩散系统的哪一层？",
    ),
    "score_sde_original": PaperGuideSpec(
        key="score_sde_original",
        part="part05_diffusion_variants",
        after_chapter=26,
        order=2,
        filename="paper_guides/02_score_sde_original.py",
        title="经典论文深读：从离散扩散到 Score-SDE 与概率流 ODE",
        paper_keys=("score_sde",),
        question="forward SDE、reverse-time SDE 与 probability flow ODE 如何共享边缘分布，却拥有不同的样本路径？",
    ),
    "flow_matching_lineage": PaperGuideSpec(
        key="flow_matching_lineage",
        part="part06_flow_matching",
        after_chapter=30,
        order=1,
        filename="paper_guides/01_cnf_to_flow_matching.py",
        title="论文谱系导读：从 CNF 的密度追踪到 Flow Matching 的速度监督",
        paper_keys=("neural_ode", "ffjord", "flow_matching"),
        question="Neural ODE、FFJORD 与 Flow Matching 分别解决了连续生成模型中的哪一个计算瓶颈？",
    ),
    "flow_matching_rectified_flow": PaperGuideSpec(
        key="flow_matching_rectified_flow",
        part="part06_flow_matching",
        after_chapter=30,
        order=2,
        filename="paper_guides/02_flow_matching_and_rectified_flow.py",
        title="经典论文深读：Flow Matching 与 Rectified Flow 到底是什么关系",
        paper_keys=("flow_matching", "rectified_flow"),
        question="两篇论文都回归条件速度，为什么 probability path、coupling、边缘速度和 reflow 仍不能混为一谈？",
    ),
}


def paper_guides_after_chapter(chapter: int) -> tuple[PaperGuideSpec, ...]:
    return tuple(
        sorted(
            (guide for guide in PAPER_GUIDES.values() if guide.after_chapter == chapter),
            key=lambda guide: guide.order,
        )
    )


def papers_for_chapter(chapter: int) -> tuple[PaperSource, ...]:
    """Return primary papers before optional tutorials, preserving catalog order."""

    return tuple(paper for paper in PAPERS.values() if chapter in paper.chapters)


PAPER_GUIDES_AFTER_CHAPTER = {
    chapter: paper_guides_after_chapter(chapter)
    for chapter in {guide.after_chapter for guide in PAPER_GUIDES.values()}
}


PAPER_GUIDES_BEFORE_CHAPTER = {
    chapter + 1: guides[-1]
    for chapter, guides in PAPER_GUIDES_AFTER_CHAPTER.items()
    if chapter < 30
}
