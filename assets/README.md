# Assets

课程专属的静态图像、图标和轻量数据放在这里。能够由 notebook 生成的图形不重复保存为人工维护资源。

## 论文原图节选

`paper_figures/` 保存课程导读实际引用的少量论文原图节选。它们只用于评论、解释和
课堂研究阅读，不能脱离对应论文来源再次分发。每张图在 notebook 中必须同时显示：

- 原论文题名、作者、版本与链接；
- 原图编号和原始图注的教学性概括；
- 颜色、坐标、线型或模块的图例说明；
- 这张图承担的学习任务；
- 该图不能证明什么。

| 本地文件 | 原始来源 |
|---|---|
| `paper_figures/vae/hinton_figure1_autoencoder.webp` | Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks*, Figure 1 |
| `paper_figures/vae/aevb_figure1_graphical_model.webp` | Kingma & Welling, *Auto-Encoding Variational Bayes*, Figure 1 |
| `paper_figures/vae/aevb_figure4_manifold.webp` | 同上，Figure 4 |
| `paper_figures/vae/vq_vae_figure1_architecture.webp` | van den Oord et al., *Neural Discrete Representation Learning*, Figure 1 |
| `paper_figures/vae/normalizing_flows_figure1.webp` | Rezende & Mohamed, *Variational Inference with Normalizing Flows*, Figure 1 |
| `paper_figures/ddim/figure1_graphical_models.webp` | Song, Meng & Ermon, *Denoising Diffusion Implicit Models*, Figure 1 |
| `paper_figures/ddim/figure4_speed_quality.webp` | 同上，Figure 4 |
| `paper_figures/ddim/ho_figure2_graphical_model.webp` | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, Figure 2 |
| `paper_figures/ddim/ho_algorithms_1_2.webp` | 同上，Algorithms 1–2 |
| `paper_figures/diffusion_variants/cfg_figure2_gaussian_guidance.webp` | Ho & Salimans, *Classifier-Free Diffusion Guidance*, Figure 2 |
| `paper_figures/diffusion_variants/ldm_figure3_architecture.webp` | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, Figure 3 |
| `paper_figures/diffusion_variants/dit_figure2_scaling.webp` | Peebles & Xie, *Scalable Diffusion Models with Transformers*, Figure 2 |
| `paper_figures/diffusion_variants/dit_figure3_architecture.webp` | 同上，Figure 3 |
| `paper_figures/diffusion_variants/score_sde_figure2_overview.webp` | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, Figure 2 |
| `paper_figures/flow_matching/neural_ode_figure1.webp` | Chen et al., *Neural Ordinary Differential Equations*, Figure 1 |
| `paper_figures/flow_matching/ffjord_figure1.webp` | Grathwohl et al., *FFJORD*, Figure 1 |
| `paper_figures/flow_matching/flow_matching_figures2_3.webp` | Lipman et al., *Flow Matching for Generative Modeling*, Figures 2–3 |
| `paper_figures/flow_matching/rectified_flow_figure2.webp` | Liu et al., *Flow Straight and Fast*, Figure 2 |

这些文件由官方 arXiv 版本裁切而来，没有改变图中数据和标记；课程只调整了裁切范围和
WebP 压缩。若论文页面更新，先核对版本和图号再替换。
