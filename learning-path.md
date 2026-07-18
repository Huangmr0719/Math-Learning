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
- [x] 所有 30 个正式章节与课程首页通过 `marimo check --strict`
- [x] 公共数学函数通过自动测试
- [x] 首页、KL 章和最小 VAE 章通过静态 HTML 导出验证
- [x] 第 14–18 章分别通过静态 HTML 执行与导出验证
- [x] 第 8–13、19–30 章分别通过静态 HTML 执行与导出验证
- [x] 完成第二轮数学与教学审查：修正 KL 支持集证明、flow 可逆性表述和 SDE 数值实现
- [x] 补齐 conditional expectation、coupling、互信息分解、importance sampling、条件 ELBO、连续性方程等数学暂停站
- [x] 建立 OU SDE、KL 支持集、条件 MSE 与 coupling marginals 的回归测试
- [x] 建立统一中英术语表，并在首页和每章首次出现处自动展示
- [x] 建立“一部分可含多篇”的经典论文导读机制，并完成自编码器谱系、AEVB 深读与 VAE 变体研讨
- [x] 为第 1–13 章建立论文坐标：中文题名、原题、版本、阅读式号/图与结论边界
- [x] 为第 14–18 章建立论文坐标，并完成扩散谱系导读与 DDPM 原文深读
- [x] 为第 19–26 章建立论文坐标，并完成 DDIM、扩散变体与 Score-SDE 导读
- [x] 为第 27–30 章建立论文坐标，并完成 CNF→Flow Matching 谱系与 Rectified Flow 深读
- [x] 建立原文短引与原图证据卡：强制说明图例、学习任务和证据边界
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
| 论文谱系导读 | `paper_guides/01_autoencoder_lineage.py` | AE、DAE、CAE 与 VAE 的数学承诺有何不同？ | 完成 |
| AEVB 深读 | `paper_guides/02_auto_encoding_variational_bayes.py` | 能否从 AEVB 原文重新构造 VAE？ | 完成 |

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
→ 对照 AE / DAE / CAE，辨认概率生成模型的分界线
→ 回到 AEVB 原文核对问题、式号、算法与实验
→ 带着标准 ELBO 进入 VAE 变体
```

两篇论文导读不占用正式章号。它们依次位于第 7 章与第 8 章之间，用原始论文训练研究阅读能力，
包括中文意译、逐式推导、图形重绘、代码映射、数值核验、证据强度判断和检索练习。

## 第二部分：VAE 变体

| 章节 | 文件 | 状态 |
|---|---|---|
| 8 | `08_beta_vae.py` | 精写 |
| 9 | `09_posterior_collapse.py` | 精写 |
| 10 | `10_iwae.py` | 精写 |
| 11 | `11_conditional_vae.py` | 精写 |
| 12 | `12_vq_vae.py` | 精写 |
| 13 | `13_flexible_posterior.py` | 精写 |
| 论文研讨 | `paper_guides/01_vae_variants_landmarks.py` | 完成 |

本部分从标准 VAE 的六类限制自然展开：目标权衡、后验失效、下界松弛、条件控制、
离散表示和后验表达能力不足，并最终用变量变换公式连接到更一般的生成过程。部分末
研讨以九篇代表论文复盘“修改了模型哪一部分”，同时把 TD-VAE 放到完成第 14 章后的
时序拓展阅读，避免提前引入尚未学习的 Markov chain 与 belief state。

## 第三部分：扩散模型与 DDPM

| 章节 | 文件 | 状态 |
|---|---|---|
| 14 | `14_iterative_denoising.py` | 精写 |
| 15 | `15_ddpm_forward.py` | 精写 |
| 16 | `16_ddpm_posterior_and_reverse.py` | 精写 |
| 17 | `17_diffusion_elbo_and_noise_prediction.py` | 精写 |
| 18 | `18_minimal_ddpm.py` | 精写 |
| 论文谱系导读 | `paper_guides/01_diffusion_lineage.py` | 扩散链与得分场为什么会在 DDPM 会合？ | 完成 |
| DDPM 深读 | `paper_guides/02_ddpm_original.py` | 能否从式 (1)–(14) 与算法 1–2 重建 DDPM？ | 完成 |

逻辑链：

```text
一次生成过难
→ 拆成马尔可夫小步骤
→ 推导任意时刻的直接加噪
→ 推导高斯反向后验
→ 把扩散模型 ELBO 化成噪声预测
→ 组成最小 DDPM
→ 回看扩散链与得分匹配两条研究路线
→ 按 DDPM 原文核验闭式、后验、目标和算法
→ 追问同一预测场是否只能沿随机 Markov 链采样
```

两篇导读依次位于第 18 章与第 19 章之间。第一篇以 Sohl-Dickstein et al.、
Hyvärinen、Vincent、Song & Ermon 和 Ho et al. 为坐标，严格区分条件 score 与
边缘 score；第二篇围绕 DDPM 原文的式号和算法，特别检查
`L_simple` 与完整 negative ELBO 不可直接画等号。由此自然进入 DDIM 的采样路径问题。

## 第四部分：DDIM

| 章节 | 文件 | 状态 |
|---|---|---|
| 19 | `19_ddim_sampling.py` | 精写 |
| 20 | `20_ddim_inversion.py` | 精写 |
| DDIM 深读 | `paper_guides/01_ddim_original.py` | 为什么同一训练目标允许更短、可确定化的路径？ | 完成 |

从“训练目标固定但采样路径未必唯一”出发，进入确定性采样、加速和反演。部分末导读
对照 DDIM 原文 Figure 1 与 Figure 4，分别区分图模型定义证据和速度—质量实验，并把
Null-text Inversion 明确归为后续真实图像编辑方法。

## 第五部分：扩散模型变体

| 章节 | 文件 | 状态 |
|---|---|---|
| 21 | `21_prediction_parameterizations.py` | 精写 |
| 22 | `22_classifier_free_guidance.py` | 精写 |
| 23 | `23_latent_diffusion.py` | 精写 |
| 24 | `24_unet_to_dit.py` | 精写 |
| 25 | `25_numerical_solvers.py` | 精写 |
| 26 | `26_score_sde_and_probability_flow_ode.py` | 精写 |
| 论文研讨 | `paper_guides/01_diffusion_variants_landmarks.py` | 六个设计旋钮分别改了系统哪一层？ | 完成 |
| Score-SDE 深读 | `paper_guides/02_score_sde_original.py` | SDE 与 ODE 为什么同边缘、不同轨迹？ | 完成 |

本部分依次回答预测什么、如何控制、在哪里扩散、用什么网络、怎样快速求解，
最后把离散扩散提升到连续时间 ODE/SDE。第一篇导读用 CFG、LDM、DiT 等原图训练
“先读图例、再下结论”；第二篇从 Fokker–Planck 与 continuity equation 严格推导
probability flow ODE 的半系数，并用一维可解实验区分粒子路径与边缘密度。

## 第六部分：流匹配（Flow Matching）

| 章节 | 文件 | 状态 |
|---|---|---|
| 27 | `27_continuous_normalizing_flow.py` | 精写 |
| 28 | `28_flow_matching.py` | 精写 |
| 29 | `29_optimal_transport_and_rectified_flow.py` | 精写 |
| 30 | `30_unified_view.py` | 精写 |
| 论文谱系导读 | `paper_guides/01_cnf_to_flow_matching.py` | CNF 的密度追踪为何转向速度监督？ | 完成 |
| FM/RF 深读 | `paper_guides/02_flow_matching_and_rectified_flow.py` | 条件路径、边缘速度、coupling 与 reflow 如何区分？ | 完成 |

逻辑链：

```text
ODE 可以运输分布
→ CNF 需要追踪密度
→ 流匹配直接学习速度场
→ 耦合（coupling）决定路径是否弯曲
→ 最优传输（OT）与 Rectified Flow 改善路径
→ 统一比较 VAE、扩散模型与流匹配
→ 回看 Neural ODE / FFJORD / Flow Matching 的计算瓶颈谱系
→ 用原文 Figure 2–3 区分条件路径与边缘速度
→ 用 Rectified Flow Figure 2 解释 crossing、rewiring 与 reflow
```

第一篇导读把“连续动力学表示、Jacobian trace、训练内 ODE 模拟”拆成三个不同问题，
并用 Hutchinson estimator 数值实验核验无偏与低方差不是同一概念。第二篇从
continuity equation 和条件期望证明 CFM 的边缘化与梯度等价，再严格区分
Flow Matching 的一般框架、线性 conditional path、Rectified Flow 与 reflow。

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
