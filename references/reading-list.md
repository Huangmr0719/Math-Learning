# 生成模型学习资源

## 本地文件

- [Auto-Encoding Variational Bayes](papers/vae/auto-encoding-variational-bayes.pdf)
- [From Autoencoder to Beta-VAE](papers/vae/from-autoencoder-to-beta-vae.pdf)
- [An Introduction to Variational Autoencoders](papers/vae/an-introduction-to-variational-autoencoders.pdf)
- [Tutorial on Variational Autoencoders](papers/vae/tutorial-on-variational-autoencoders.pdf)
- [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)

本地文件便于离线阅读；下面的官方链接用于确认版本、作者和引用信息。

## Knowledge

### VAE 论文谱系：先区分“重构正则化”与“概率生成”

- [Paper: Reducing the Dimensionality of Data with Neural Networks — Hinton & Salakhutdinov](https://www.science.org/doi/10.1126/science.1127647)
  深层自编码器的历史起点。第 1 章与谱系导读用于检查压缩、重构与生成之间的缺口。
- [Paper: Extracting and Composing Robust Features with Denoising Autoencoders — Vincent et al.](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
  去噪自编码器原始论文。用于区分“损坏输入的训练噪声”与“概率模型中的潜变量”。
- [Paper: Contractive Auto-Encoders — Rifai et al.](https://icml.cc/2011/papers/455_icmlpaper.pdf)
  收缩自编码器原始论文。用于理解 encoder Jacobian 正则、局部不变性及其结论边界。
- [Paper: Auto-Encoding Variational Bayes — Kingma & Welling](https://arxiv.org/abs/1312.6114)
  VAE 原始论文。用于核对变分下界、重参数化估计器与基本概率模型。
- [Tutorial: An Introduction to Variational Autoencoders — Kingma & Welling](https://arxiv.org/abs/1906.02691)
  系统 VAE 教程。用于概率图、ELBO、后验扩展和深层生成模型。
- [Tutorial on Variational Autoencoders — Carl Doersch](https://arxiv.org/abs/1606.05908)
  偏直觉和计算机视觉视角。用于为初学者解释 latent variable、ELBO 与 CVAE。
- [Blog map: From Autoencoder to Beta-VAE — Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)
  二级教学资料，用于发现论文谱系，不作为公式或历史主张的最终依据。课程对其中的
  约束等价、VQ-VAE 码本更新等表述补充了适用条件。

### VAE 变体：六类问题与代表论文

- [Paper: beta-VAE — Higgins et al.](https://openreview.net/forum?id=Sy2fzU9gl)
  用于第 8 章：KL 权重、表示容量、潜变量遍历与解耦实验。
- [Paper: Understanding Disentangling in beta-VAE — Burgess et al.](https://arxiv.org/abs/1804.03599)
  用于第 8 章：率失真视角、目标容量 (C) 与渐进容量调度。该工作为 2018 arXiv
  版本；不要沿用博客参考文献中含混的 “NIPS 2017” 标注。
- [Paper: Generating Sentences from a Continuous Space — Bowman et al.](https://arxiv.org/abs/1511.06349)
  用于第 9 章：强自回归 decoder 下的训练困难、KL 退火与 posterior collapse 早期案例。
- [Paper: Lagging Inference Networks and Posterior Collapse in VAEs — He et al.](https://openreview.net/forum?id=rylDfnCqF7)
  用于第 9 章：inference lag 的训练动态解释与 aggressive inference updates。
- [Paper: Importance Weighted Autoencoders — Burda et al.](https://arxiv.org/abs/1509.00519)
  用于第 10 章：importance weights、K 样本下界、单调趋紧条件与数值稳定实现。
- [Paper: Learning Structured Output Representation using Deep Conditional Generative Models — Sohn et al.](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
  用于第 11 章：CVAE 条件概率图、训练识别网络与测试条件 prior 的两条路径。
- [Paper: Neural Discrete Representation Learning — van den Oord et al.](https://papers.nips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html)
  用于第 12 章：VQ-VAE、codebook、commitment loss、straight-through 与离散 prior。
- [Paper: Generating Diverse High-Fidelity Images with VQ-VAE-2 — Razavi et al.](https://arxiv.org/abs/1906.00446)
  第 12 章拓展：层次 codebook、两阶段训练和强自回归先验；不把质量提升只归因于离散化。
- [Paper: Variational Inference with Normalizing Flows — Rezende & Mohamed](https://arxiv.org/abs/1505.05770)
  用于第 13 章：可逆变换、变量替换、Jacobian log-determinant 与灵活近似后验。
- [Paper: Temporal Difference Variational Auto-Encoder — Gregor et al.](https://arxiv.org/abs/1806.03107)
  拓展阅读：belief state 与 jumpy prediction。建议完成第 14 章 Markov chain 后阅读，
  不作为当前第 8–13 章主线的先修要求。

### Diffusion、DDIM 与 Flow Matching

- [Paper: Denoising Diffusion Probabilistic Models — Ho et al.](https://arxiv.org/abs/2006.11239)
  DDPM 核心来源。用于前向过程、反向参数化和简化噪声预测目标。
- [Paper: Deep Unsupervised Learning using Nonequilibrium Thermodynamics — Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585)
  早期 diffusion probabilistic model 原始来源。用于理解逐步破坏、反向过程与变分训练的历史起点。
- [Paper: Estimation of Non-Normalized Statistical Models by Score Matching — Hyvärinen](https://www.jmlr.org/papers/v6/hyvarinen05a.html)
  得分匹配的基础来源。用于区分密度值与对数密度梯度，并核对分部积分所需的正则条件。
- [Paper: A Connection Between Score Matching and Denoising Autoencoders — Vincent](https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and)
  去噪得分匹配的基础来源。用于解释带噪条件 score、边缘 score 和去噪目标之间的关系。
- [Paper: Generative Modeling by Estimating Gradients of the Data Distribution — Song & Ermon](https://arxiv.org/abs/1907.05600)
  噪声条件得分网络（NCSN）的原始来源。用于多噪声尺度得分学习与退火 Langevin dynamics。
- [Paper: Improved Denoising Diffusion Probabilistic Models — Nichol & Dhariwal](https://proceedings.mlr.press/v139/nichol21a.html)
  DDPM 后续改进。用于学习反向方差、混合目标、采样步数与 likelihood/样本质量的区分；
  不把这些后续结果倒写成 2020 年 DDPM 原论文的贡献。
- [Paper: Denoising Diffusion Implicit Models — Song et al.](https://arxiv.org/abs/2010.02502)
  DDIM 核心来源。用于非 Markov 过程、确定性采样与加速。
- [Paper: Null-text Inversion — Mokady et al.](https://arxiv.org/abs/2211.09794)
  DDIM inversion 的后续真实图像编辑方法。用于理解大 CFG 下的重构误差和无条件文本嵌入优化。
- [Paper: Progressive Distillation for Fast Sampling of Diffusion Models — Salimans & Ho](https://arxiv.org/abs/2202.00512)
  用于 velocity parameterization 与逐轮减半采样步数；参数化改善应与蒸馏程序一起理解。
- [Paper: Elucidating the Design Space of Diffusion-Based Generative Models — Karras et al.](https://arxiv.org/abs/2206.00364)
  EDM 的统一设计坐标。用于拆分预条件、噪声分布、训练权重、采样轨迹和 solver。
- [Paper: Diffusion Models Beat GANs on Image Synthesis — Dhariwal & Nichol](https://arxiv.org/abs/2105.05233)
  分类器引导的主要来源；需要额外训练能处理 noisy inputs 的分类器。
- [Paper: Classifier-Free Diffusion Guidance — Ho & Salimans](https://arxiv.org/abs/2207.12598)
  CFG 的主要来源。用于有条件/无条件 score 组合及质量—多样性折中。
- [Paper: Score-Based Generative Modeling through SDEs — Song et al.](https://arxiv.org/abs/2011.13456)
  用于连接 score、reverse-time SDE 与 probability flow ODE。
- [Paper: Neural Ordinary Differential Equations — Chen et al.](https://arxiv.org/abs/1806.07366)
  CNF 的连续时间起点。用于核对神经 ODE、瞬时变量替换、Jacobian trace 与解唯一性条件。
- [Paper: FFJORD — Grathwohl et al.](https://arxiv.org/abs/1810.01367)
  用 Hutchinson estimator 近似 CNF 的 Jacobian trace；阅读时区分 probe 无偏性、估计方差与 ODE 离散误差。
- [Paper: Flow Matching for Generative Modeling — Lipman et al.](https://arxiv.org/abs/2210.02747)
  Flow Matching 主来源。用于 FM/CFM 目标、条件速度的边缘化、梯度等价、diffusion path 和 OT path。
- [Paper: High-Resolution Image Synthesis with Latent Diffusion Models — Rombach et al.](https://arxiv.org/abs/2112.10752)
  Latent Diffusion 的压缩空间、条件机制与感知压缩来源。
- [Paper: Scalable Diffusion Models with Transformers — Peebles & Xie](https://arxiv.org/abs/2212.09748)
  DiT 架构、patch token 和 scaling 分析来源。
- [Paper: U-Net — Ronneberger et al.](https://arxiv.org/abs/1505.04597)
  U-Net 原始架构来源。扩散 U-Net 借用多尺度路径和 skip connection，但任务和模块已经变化。
- [Paper: DPM-Solver — Lu et al.](https://arxiv.org/abs/2206.00927)
  扩散 ODE 专用高阶求解器来源。比较速度时以 NFE、时间网格和同一模型为共同条件。
- [Paper: Flow Straight and Fast — Liu et al.](https://arxiv.org/abs/2209.03003)
  Rectified Flow、rewiring、reflow 与凸运输成本不增的主要来源；不能简写为“一次训练精确求出 OT”。
- [Documentation: marimo](https://docs.marimo.io/)
  交互 notebook 的官方来源。用于反应式执行、UI、绘图、缓存和导出。
- [Course: Hugging Face Diffusion Models Class](https://github.com/huggingface/diffusion-models-class)
  Diffusers、从零实现、guidance、Stable Diffusion 和 DDIM inversion 的工程参考。

## Wisdom (Communities)

- [Hugging Face Forums](https://discuss.huggingface.co/)
  适合验证实现问题、训练配置和 Diffusers 生态中的工程经验。
- [PyTorch Forums](https://discuss.pytorch.org/)
  适合检查 autograd、shape、数值稳定性和设备相关问题。

## Gaps

- 第二轮精修时补充 EDM、consistency/distillation 与现代 flow/diffusion 统一参数化资料。
- 为不同 Rectified Flow 版本建立专门的记号对照表，避免把 reflow、OT coupling 与一般 CFM 混用。
