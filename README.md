# Math-Learning：从变分自编码器（VAE）到流匹配（Flow Matching）

一套面向高中数学基础学习者的生成模型交互式课程。课程正文使用中文，
专业术语第一次出现时给出通用中文译名、英文原词和简明解释；公式、代码变量、
常用缩写与论文标题保留英文。marimo notebook 是唯一维护的教学正文。

## 学习主线

```text
自编码器与潜在表示（Autoencoder / latent representation）
→ VAE 概率建模
→ VAE 变体
→ 扩散模型 / DDPM
→ DDIM
→ 扩散模型变体与连续时间视角
→ 连续归一化流（CNF）
→ 流匹配（Flow Matching）
→ 最优传输（OT）与 Rectified Flow
```

数学不被集中堆放在课程开头。每章遵循同一条学习循环：

```text
提出模型问题
→ 发现当前缺少的数学工具
→ 从直觉、数字和图形建立理解
→ 给出严格定义与推导
→ 对照代码和张量形状（tensor shape）
→ 数值验证与交互可视化
→ 错误案例与分层练习
→ 引出下一章
```

完整章节路线、完成状态和跨章节逻辑见
[learning-path.md](learning-path.md)。

## 快速开始

环境要求：

- Python 3.11 或更高版本
- marimo 0.23.x
- NumPy、Matplotlib、PyTorch 等依赖，详见 `pyproject.toml`

如需创建独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

启动包含全部章节的 marimo 目录工作区：

```bash
marimo edit notebooks
```

浏览器打开后，先选择 `00_home.py` 进入课程首页。课程链接在运行时解析为
当前仓库中的 Notebook 路径，因此也兼容直接运行
`marimo edit notebooks/00_home.py`；不过目录工作区更方便浏览全部章节。

第一次学习建议从第 1 章开始。首页提供六个部分的课程地图、章节状态、
统一符号索引、专业术语中英对照和跨模型符号切换说明。每章正文开始前还会
显示该章首次引入的术语。章节链接会在新标签页中打开，以便保留课程地图。

## 目录结构

```text
Math-Learning/
├── README.md                    # 项目入口
├── learning-path.md             # 课程路线、章节逻辑与建设状态
├── pyproject.toml               # Python 与科学计算依赖
├── AGENTS.md                    # 项目维护约定
│
├── notebooks/                   # 唯一维护的教学正文
│   ├── 00_home.py               # 课程首页
│   ├── part01_vae/              # 第 1–7 章与部分末经典论文导读
│   │   └── paper_guides/        # 自编码器谱系与 AEVB 原文深读
│   ├── part02_vae_variants/     # 第 8–13 章与变体论文研讨
│   │   └── paper_guides/        # 六类缺口、九篇代表论文
│   ├── part03_diffusion_ddpm/   # 第 14–18 章与扩散/DDPM 论文导读
│   │   └── paper_guides/        # 扩散谱系与 DDPM 原文深读
│   ├── part04_ddim/             # 第 19–20 章与 DDIM 原文深读
│   │   └── paper_guides/        # DDIM 路径、速度证据与 inversion 边界
│   ├── part05_diffusion_variants/ # 第 21–26 章与两篇论文导读
│   │   └── paper_guides/        # 六层设计研讨与 Score-SDE 原文深读
│   └── part06_flow_matching/    # 第 27–30 章与两篇论文导读
│       └── paper_guides/        # CNF 谱系、Flow Matching 与 Rectified Flow 深读
│
├── src/
│   ├── teaching/                # 章节目录和公共教学组件
│   ├── math_checks/             # 可复用数学与数值验证
│   ├── models/                  # 教学章节复用的模型实现
│   └── visualization/           # 统一绘图风格和缓存设置
│
├── tests/                       # 数学公式、课程结构和防回归测试
├── scripts/                     # 辅助检查脚本
│
├── references/
│   ├── papers/                  # 本地论文与教程，按主题归档
│   ├── guides/                  # 数学审查清单、术语规范和背景路线
│   └── reading-list.md          # 权威链接与阅读说明
│
├── docs/
│   ├── plans/                   # 学习计划与任务清单
│   └── learning-profile/        # 学习目标、偏好与学习记录
│
├── assets/                      # 不适合由 notebook 动态生成的静态资源
└── exports/                     # marimo 静态 HTML 发布产物
```

## 六个课程部分

| 部分 | 章节 | 核心问题 |
|---|---:|---|
| 从表示学习到 VAE | 1–7 | 怎样把重构模型变成可采样的概率生成模型？ |
| VAE 变体 | 8–13 | 怎样改变容量、后验、条件和 latent 类型？ |
| Diffusion 与 DDPM | 14–18 | 怎样通过逐步加噪和逆向去噪生成数据？ |
| DDIM | 19–20 | 同一训练网络为什么可以采用更快、确定性的采样路径？ |
| Diffusion 变体 | 21–26 | 预测什么、如何控制、在哪里扩散、怎样连续化？ |
| Flow Matching | 27–30 | 怎样直接学习把噪声运输到数据的速度场？ |

每个课程部分结束后可设置一个或多个不占用正式章号的“经典论文导读”。导读不是论文摘要，
而是把原文问题、中文意译、关键推导、图形解释、代码映射、数值验证与批判性阅读
组织成 marimo 专题站。VAE 与 VAE 变体部分现已形成三站论文路线：

```text
AE / DAE / CAE 与 VAE 的思想边界
→ AEVB 原文逐式深读
→ β-VAE、坍塌、IWAE、CVAE、VQ-VAE、Flow 的多论文研讨
```

Diffusion/DDPM 部分形成两站路线：

```text
扩散概率链与得分匹配的历史、数学会合
→ DDPM 原文式 (1)–(14) 与算法 1–2 的逐式深读
→ 带着“采样路径是否唯一”的问题进入 DDIM
```

DDIM 与扩散变体继续形成三站路线：

```text
DDIM 原文：非马尔可夫路径、速度—质量与 inversion 边界
→ CFG / LDM / DiT / Solver 等六层设计研讨
→ Score-SDE 与 probability flow ODE 的同边缘、不同轨迹
→ 进入 Flow Matching，直接学习速度场
```

Flow Matching 部分以两站收束整门课程：

```text
Neural ODE → FFJORD → Flow Matching：表示、trace 与训练信号的三次变化
→ Flow Matching / Rectified Flow：条件路径、边缘速度、coupling、rewiring 与 reflow
```

第 1–30 章开头还会按当前知识点显示“论文坐标”，给出通用中文译名、原题、
作者版本以及建议定点阅读的式号、图和实验。它用于学完正文后的原文复核，不要求
初学者在进入章节前先通读论文。

## 教学内容与参考资料的边界

- `notebooks/` 是唯一需要同步维护的教学正文。
- `src/` 保存可复用实现，不隐藏章节中的关键推导。
- `references/` 保存本地论文、权威外部链接和阅读指南，不属于课程源码。
- `exports/` 是生成产物，不应手工编辑。

本地论文目录见 [references/README.md](references/README.md)。课程章节中的公式应以
原始论文、严格推导和可执行验证共同核对，不能仅依据旧笔记或单次数值实验。
术语翻译规则见 [references/guides/terminology.md](references/guides/terminology.md)。

## 验证

每次修改章节或公共数学实现后运行：

```bash
marimo check --strict notebooks
python -m pytest -q
python -m compileall -q src notebooks
```

快速运行可复用数学 demo：

```bash
python scripts/math_check.py --demo all
```

导出一个带执行结果的静态章节：

```bash
marimo export html notebooks/part01_vae/04_kl_divergence.py \
  -o exports/04_kl_divergence.html
```

课程禁止在打开 notebook 时自动下载数据或启动昂贵训练。需要下载、训练或加载
大型模型的操作必须由显式按钮触发。

`tests/test_marimo_features.py` 还会检查每个正式章节是否真正使用 marimo：

- 至少有一个实际显示、并驱动下游 cell 的本章专属 UI；
- 包含响应式可视化、直觉/严格双栏 tabs 和折叠练习答案；
- 不允许在同一 cell 中放置多个未组合的展示表达式，避免标题或推导地图静默消失；
- 不允许正式章节退化为 `chapter_scaffold` 通用骨架。

## 当前状态

- 课程首页和 30 个正式章节已经建立。
- 六个部分均完成首轮精写。
- 已完成第二轮数学与教学审查。
- 关键数学主张具有自动测试或 notebook 数值验证。
- 30 个正式章节已逐一完成带执行结果的 HTML 导出与 marimo 特性验收。
- 第 1–30 章已接入 VAE、Diffusion 与 Flow Matching 论文坐标，并完成十个部分末论文导读。
- 论文导读采用原文短引和少量原图节选；每张图必须解释图例、学习任务和证据边界。
- 第一部分已完成自编码器谱系与 *Auto-Encoding Variational Bayes* 两个交互式导读。
- 第二部分已完成覆盖九篇代表论文的 VAE 变体交互式研讨。
- 正式 HTML 批量发布仍属于后续工作。

## 阅读建议

学习章节时优先保持主线连续，不要先阅读所有论文。遇到某章需要进一步研究时，
再从 [references/reading-list.md](references/reading-list.md) 选择对应原始来源。
