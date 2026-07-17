# 基于 marimo 的生成模型学习路径

## 核心原则

数学不再集中放在课程开头，而是在模型真正需要它时出现：

```text
遇到模型问题
→ 发现缺少数学工具
→ 从高中知识出发学习
→ 立即用于推导
→ 代码验证
→ 可视化实验
→ 错误案例
→ 引出下一章
```

marimo `.py` 文件是唯一教学源。旧的 Markdown 推导继续作为迁移参考，不再与
notebook 同时维护两份正文。

专业术语遵循“中文先行、英文对照”：首次出现时给出中文译名、英文原词和
一句话解释，后续正文优先使用中文。完整规则见
`references/guides/terminology.md`。

## 当前建设状态

- [x] 建立 `pyproject.toml`、六个部分目录、首页和全部 30 个章节文件
- [x] 建立统一章节目录、教学组件、数学检查、可视化样式和模型模块
- [x] 完成第一部分 VAE（第 1–7 章）的首轮精写
- [x] 完成第二部分 VAE 变体（第 8–13 章）的首轮精写
- [x] 完成第三部分 Diffusion/DDPM（第 14–18 章）的首轮精写
- [x] 完成第四部分 DDIM（第 19–20 章）的首轮精写
- [x] 完成第五部分 Diffusion 变体（第 21–26 章）的首轮精写
- [x] 完成第六部分 Flow Matching（第 27–30 章）的首轮精写
- [x] 所有 31 个 notebook 通过 `marimo check --strict`
- [x] 公共数学函数通过自动测试
- [x] 首页、KL 章和最小 VAE 章通过静态 HTML 导出验证
- [x] 第 14–18 章分别通过静态 HTML 执行与导出验证
- [x] 第 8–13、19–30 章分别通过静态 HTML 执行与导出验证
- [x] 完成第二轮数学与教学审查：修正 KL 支持集证明、flow 可逆性表述和 SDE 数值实现
- [x] 补齐 conditional expectation、coupling、互信息分解、importance sampling、条件 ELBO、连续性方程等数学暂停站
- [x] 建立 OU SDE、KL 支持集、条件 MSE 与 coupling marginals 的回归测试
- [x] 建立统一中英术语表，并在首页和每章首次出现处自动展示
- [ ] 将全部章节的最终 HTML 批量写入 `exports/` 发布目录

状态含义：

- `精写`：已经具备完整教学叙事、交互、验证、反例和分层练习。
- `骨架`：已经具备元数据、先修关系、统一章节协议和下一章桥梁，等待逐章精写。

## 第一部分：从表示学习到变分自编码器（VAE）

| 章节 | 文件 | 关键问题 | 状态 |
|---|---|---|---|
| 1 | `01_autoencoder_and_latent.py` | 压缩和重构是否等于学会生成？ | 精写 |
| 2 | `02_latent_distribution.py` | 如何让潜空间连续、可采样并具有概率意义？ | 精写 |
| 3 | `03_bayesian_posterior.py` | 给定观测 \(x\)，怎样反推产生它的 \(z\)？ | 精写 |
| 4 | `04_kl_divergence.py` | 怎样衡量近似分布与目标分布的差异？ | 精写 |
| 5 | `05_elbo.py` | 无法直接最大化 \(\log p(x)\) 时怎么办？ | 精写 |
| 6 | `06_gaussian_and_reparameterization.py` | 怎样让随机采样参与反向传播？ | 精写 |
| 7 | `07_minimal_vae.py` | 如何把概率图、ELBO、网络和实验闭环？ | 精写 |

逻辑链：

```text
自编码器（Autoencoder）能重构
→ 潜空间（latent space）却不一定可采样
→ 把潜在点改成条件分布
→ 用贝叶斯后验表达反推问题
→ 用 KL 衡量近似误差
→ 用 ELBO 得到可训练下界
→ 用高斯分布与重参数化传递梯度
→ 组成最小 VAE
```

## 第二部分：VAE 变体

| 章节 | 文件 | 状态 |
|---|---|---|
| 8 | `08_beta_vae.py` | 精写 |
| 9 | `09_posterior_collapse.py` | 精写 |
| 10 | `10_iwae.py` | 精写 |
| 11 | `11_conditional_vae.py` | 精写 |
| 12 | `12_vq_vae.py` | 精写 |
| 13 | `13_flexible_posterior.py` | 精写 |

本部分从标准 VAE 的三个限制自然展开：目标权衡、后验失效和后验
表达能力不足，并最终用变量变换公式连接到更一般的生成过程。

## 第三部分：扩散模型与 DDPM

| 章节 | 文件 | 状态 |
|---|---|---|
| 14 | `14_iterative_denoising.py` | 精写 |
| 15 | `15_ddpm_forward.py` | 精写 |
| 16 | `16_ddpm_posterior_and_reverse.py` | 精写 |
| 17 | `17_diffusion_elbo_and_noise_prediction.py` | 精写 |
| 18 | `18_minimal_ddpm.py` | 精写 |

逻辑链：

```text
一次生成过难
→ 拆成马尔可夫小步骤
→ 推导任意时刻的直接加噪
→ 推导高斯反向后验
→ 把扩散模型 ELBO 化成噪声预测
→ 组成最小 DDPM
```

## 第四部分：DDIM

| 章节 | 文件 | 状态 |
|---|---|---|
| 19 | `19_ddim_sampling.py` | 精写 |
| 20 | `20_ddim_inversion.py` | 精写 |

从“训练目标固定但采样路径未必唯一”出发，进入确定性采样、加速和反演。

## 第五部分：扩散模型变体

| 章节 | 文件 | 状态 |
|---|---|---|
| 21 | `21_prediction_parameterizations.py` | 精写 |
| 22 | `22_classifier_free_guidance.py` | 精写 |
| 23 | `23_latent_diffusion.py` | 精写 |
| 24 | `24_unet_to_dit.py` | 精写 |
| 25 | `25_numerical_solvers.py` | 精写 |
| 26 | `26_score_sde_and_probability_flow_ode.py` | 精写 |

本部分依次回答预测什么、如何控制、在哪里扩散、用什么网络、怎样快速求解，
最后把离散扩散提升到连续时间 ODE/SDE。

## 第六部分：流匹配（Flow Matching）

| 章节 | 文件 | 状态 |
|---|---|---|
| 27 | `27_continuous_normalizing_flow.py` | 精写 |
| 28 | `28_flow_matching.py` | 精写 |
| 29 | `29_optimal_transport_and_rectified_flow.py` | 精写 |
| 30 | `30_unified_view.py` | 精写 |

逻辑链：

```text
ODE 可以运输分布
→ CNF 需要追踪密度
→ 流匹配直接学习速度场
→ 耦合（coupling）决定路径是否弯曲
→ 最优传输（OT）与 Rectified Flow 改善路径
→ 统一比较 VAE、扩散模型与流匹配
```

## 旧材料的迁移策略

以下内容保留，不删除：

- `docs/plans/VAE_学习_ToDO_清单.md`：原始学习记录与消融计划。
- `docs/learning-profile/`：学习目标、教学偏好和学习记录。
- `references/papers/`：按主题集中保存的本地论文。
- `references/reading-list.md`：外部论文、文档和课程的权威链接。

旧 Markdown 推导已经完成迁移并删除，避免与 notebook 并行维护两份正文。
新增内容应先核对原始资料和数值验证，再直接写入对应 notebook。

## 运行与验收

```bash
# 启动全部章节所在的目录工作区，再从文件列表打开 00_home.py
marimo edit notebooks

# 严格检查全部 notebook
marimo check --strict notebooks

# 测试公共数学函数
python -m pytest -q

# 导出一章静态 HTML
marimo export html notebooks/part01_vae/04_kl_divergence.py \
  -o exports/04_kl_divergence.html
```
