# 生成模型学习资源

## 本地文件

- [From Autoencoder to Beta-VAE](papers/vae/from-autoencoder-to-beta-vae.pdf)
- [An Introduction to Variational Autoencoders](papers/vae/an-introduction-to-variational-autoencoders.pdf)
- [Tutorial on Variational Autoencoders](papers/vae/tutorial-on-variational-autoencoders.pdf)
- [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)

本地文件便于离线阅读；下面的官方链接用于确认版本、作者和引用信息。

## Knowledge

- [Paper: Auto-Encoding Variational Bayes — Kingma & Welling](https://arxiv.org/abs/1312.6114)
  VAE 原始论文。用于核对变分下界、重参数化估计器与基本概率模型。
- [Tutorial: An Introduction to Variational Autoencoders — Kingma & Welling](https://arxiv.org/abs/1906.02691)
  系统 VAE 教程。用于概率图、ELBO、后验扩展和深层生成模型。
- [Tutorial on Variational Autoencoders — Carl Doersch](https://arxiv.org/abs/1606.05908)
  偏直觉和计算机视觉视角。用于为初学者解释 latent variable、ELBO 与 CVAE。
- [Paper: Denoising Diffusion Probabilistic Models — Ho et al.](https://arxiv.org/abs/2006.11239)
  DDPM 核心来源。用于前向过程、反向参数化和简化噪声预测目标。
- [Paper: Deep Unsupervised Learning using Nonequilibrium Thermodynamics — Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585)
  早期 diffusion probabilistic model 原始来源。用于理解逐步破坏、反向过程与变分训练的历史起点。
- [Paper: Denoising Diffusion Implicit Models — Song et al.](https://arxiv.org/abs/2010.02502)
  DDIM 核心来源。用于非 Markov 过程、确定性采样与加速。
- [Paper: Score-Based Generative Modeling through SDEs — Song et al.](https://arxiv.org/abs/2011.13456)
  用于连接 score、reverse-time SDE 与 probability flow ODE。
- [Paper: Flow Matching for Generative Modeling — Lipman et al.](https://arxiv.org/abs/2210.02747)
  Flow Matching 主来源。用于 conditional flow matching、概率路径和 OT path。
- [Paper: Importance Weighted Autoencoders — Burda et al.](https://arxiv.org/abs/1509.00519)
  用于 IWAE 下界、importance weights 与多样本估计。
- [Paper: Neural Discrete Representation Learning — van den Oord et al.](https://arxiv.org/abs/1711.00937)
  VQ-VAE、codebook、commitment loss 与 straight-through estimator 的原始来源。
- [Paper: High-Resolution Image Synthesis with Latent Diffusion Models — Rombach et al.](https://arxiv.org/abs/2112.10752)
  Latent Diffusion 的压缩空间、条件机制与感知压缩来源。
- [Paper: Scalable Diffusion Models with Transformers — Peebles & Xie](https://arxiv.org/abs/2212.09748)
  DiT 架构、patch token 和 scaling 分析来源。
- [Paper: Flow Straight and Fast — Liu et al.](https://arxiv.org/abs/2209.03003)
  Rectified Flow、reflow 与直线 coupling 的主要来源。
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
