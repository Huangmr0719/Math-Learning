# References

本目录集中保存课程使用的外部资料。这里的内容用于核对、延伸阅读和工程参考，
不属于 30 个正式章节的教学源码。

## 目录

```text
references/
├── README.md
├── reading-list.md
├── papers/
│   └── vae/
└── guides/
    ├── math-checklist.md
    └── ml-math-roadmap.md
```

## 本地论文与教程

### VAE

| 文件 | 类型 | 用途 |
|---|---|---|
| `papers/vae/auto-encoding-variational-bayes.pdf` | Kingma & Welling 原始论文，arXiv v10 / ICLR 2014 | 第一部分论文导读；核对式 (1)–(10)、算法 1 与图 1–5 |
| `papers/vae/from-autoencoder-to-beta-vae.pdf` | Lilian Weng 教学文章 | Autoencoder、VAE、Beta-VAE 的直观串联 |
| `papers/vae/an-introduction-to-variational-autoencoders.pdf` | Kingma & Welling 系统教程 | ELBO、概率图、后验扩展与深层生成模型 |
| `papers/vae/tutorial-on-variational-autoencoders.pdf` | Carl Doersch 教程，arXiv:1606.05908 | 直觉、ELBO、重参数化、CVAE 与视觉例子 |

原目录中的 `1606.05908v3.pdf` 与
`tutorial-on-variational-autoencoders.pdf` 内容完全相同，因此只保留命名清楚的一份。

原始 VAE 论文 *Auto-Encoding Variational Bayes* 同时保留本地 v10 与
`reading-list.md` 中的 arXiv 官方入口：本地文件保证导读可离线核对，官方入口用于
确认版本和引用信息。

其余 VAE 变体与 Diffusion/DDPM 论文暂不重复保存 PDF；统一从 `reading-list.md` 的
会议、期刊、OpenReview 或 arXiv 官方入口访问。这样既能核对版本，也避免仓库因同一论文
的多个 PDF 版本持续膨胀。`src/teaching/paper_guides.py` 是章节—论文—式号/图号映射的
单一数据源。

## 外部课程

Hugging Face Diffusion Models Course 不在本仓库重复保存。需要工程参考时，通过
`reading-list.md` 中的官方 GitHub 链接访问，以避免仓库包含大体积、停止同步的
第三方课程副本。

## 阅读指南

- `reading-list.md`：按知识主题列出原始论文、教程和社区。
- `guides/math-checklist.md`：长推导审查清单。
- `guides/ml-math-roadmap.md`：本课程涉及的数学背景地图。

新增本地论文时，应放入 `papers/<topic>/`，使用可辨认的英文小写文件名，并在本页
登记标题、类型和对应章节。不要把 PDF 重新放回项目根目录。
