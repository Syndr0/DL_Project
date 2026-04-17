# 项目规划：Image Similarity Recommendation with CNN Re-ranking

## 1. 项目目标

我们要做一个**图像相似推荐系统**。  
给一张 query image，系统先找出一批最相似的候选图，再进一步优化排序结果。

整个项目分成两部分：

1. **Baseline：Top-K Image Retrieval**
2. **改进：CNN-based Lightweight Re-ranking**

---

## 2. Baseline

### 要复现的仓库
- **Top-k Image Retrieval repo**  
  https://github.com/tercasaskova311/Top-k-Image-Retrieval-Image-recognition-/tree/main  
  这个仓库实现了多个模型（CLIP、DINOv2、EfficientNet、ResNet、GoogLeNet）的 embedding 提取，并用 **cosine similarity** 做 Top-k 检索。[3]

### Baseline 的核心流程

```text
query image
→ encoder 提取 embedding
→ 用 cosine similarity 和 gallery 做相似度计算
→ 返回 Top-K 结果
```

### Baseline 的本质

Baseline 用的是一个**固定的相似度函数**：

- 输入：query embedding 和 candidate embedding
- 输出：cosine similarity score

也就是说，它不会学习“什么样才算更像”，只是用一个固定公式来排结果。

---

## 3. 参考论文

### 主要参考论文

- **STIR: Siamese Transformer for Image Retrieval Postprocessing**  
  arXiv: https://arxiv.org/abs/2304.13393 [1]

### 补充参考论文

- **Instance-Level Image Retrieval Using Reranking Transformers**  
  ICCV 2021 Open Access:  
  https://openaccess.thecvf.com/content/ICCV2021/html/Tan_Instance-Level_Image_Retrieval_Using_Reranking_Transformers_ICCV_2021_paper.html [2]

---

## 4. Related Work 要点

### 4.1 Baseline 类方法

这类方法先提取图像 embedding，再直接用 cosine similarity 排序。Top-k repo 就属于这一类。它的优点是简单、快，但缺点是排序比较粗糙，因为 cosine 只是固定规则，不会针对任务学习更细的相似性。[3]

### 4.2 Transformer-based reranking

STIR 这类方法的思路是：

1. 先做初步 retrieval，拿到 Top-K 候选
2. 再用 Transformer 比较 `(query, candidate)`，输出更精细的相似度分数
3. 根据新分数重新排序

也就是说，Transformer 不是替代 retrieval，而是作为 **postprocessing / reranking** 模块。[1]

---

## 5. 我们的方案

### 5.1 总体思路

我们不改 baseline 的 retrieval 主体，而是在后面加一个更轻量的 reranking 模块：

```text
query image
→ baseline retrieval
→ 得到 Top-K candidates
→ lightweight reranking
→ 输出优化后的排序结果
```

### 5.2 我们的创新点

我们的创新不在于“加了 rerank”本身，因为 reranking 早就有人做了。真正的点在于：

- **把固定相似度函数换成可学习的相似度函数**
- **不用 Transformer，而是用更轻量的 CNN 做 reranking**
- **在效果和计算成本之间做一个更好的 trade-off**

更直接地说：

- baseline：`cosine similarity`
- transformer 方法：`Transformer 学 similarity`
- 我们：`轻量 CNN 学 similarity`

---

## 6. 我们的方法设计（CNN 精排）

### 6.1 系统分两步

**第一步：粗排（Baseline Retrieval）**

- 输入一张 query image
- 用 baseline encoder 提取特征
- 与 gallery 特征做 cosine similarity
- 返回 Top-K 候选

这一步负责快速筛出“看起来比较像”的候选图，但排序仍较粗糙。

**第二步：精排（CNN Reranking）**

- 不直接使用 baseline 的原顺序
- 对 Top-K 中每个 candidate 与 query 组成图像对
- 用轻量 CNN 计算 pairwise similarity score
- 按新分数重新排序

### 6.2 CNN rerank 的输入与输出

输入：

- query image
- candidate image

输出：

- 一个实数相似度分数（例如 0.95 / 0.72 / 0.20）

该分数用于对 Top-K 候选进行重新排序。

### 6.3 CNN 如何比较两张图

采用最直观的 pairwise 输入方式：

- 将 query 和 candidate 在通道维拼接
- 单张 RGB 图是 3 通道
- 拼接后输入为 6 通道张量（前 3 通道 query，后 3 通道 candidate）

然后 CNN 学习判断：

- 形状是否相似
- 局部纹理是否相似
- 颜色差异是否关键
- 哪些差异是噪声、哪些差异是语义差异

### 6.4 CNN rerank 具体流程

1. Baseline 先返回 Top-K（如 Top-20）
2. 构造图像对：`(query, candidate_i)`
3. 每一对拼接为 6 通道输入并送入 CNN
4. 得到 `score(query, candidate_i)`
5. 按分数降序重排并输出最终结果

### 6.5 CNN 的训练方式

训练样本形式：

- `(query, candidate, label)`
- `label=1` 表示相似，`label=0` 表示不相似

若数据集有类别标签，可用同类构造正样本、异类构造负样本。训练目标是让 CNN 学会：

- 相似对输出高分
- 不相似对输出低分

可将输出视为 0~1 概率，使用二分类损失训练。

### 6.6 为什么选择 CNN rerank

- baseline 已经完成大规模粗筛，精排只需处理 Top-K，计算压力可控
- 小 CNN 能体现 learned similarity 的核心思想
- 相比 Transformer rerank 更轻量，更适合课程项目与低算力环境

---

## 7. 项目中的几个方法角色

### 7.1 Baseline

- Top-k repo
- embedding + cosine similarity
- 用来提供初步检索结果

### 7.2 文献方法

- STIR / Reranking Transformer
- 用 Transformer 对 Top-K 做更精细比较
- 作为我们方法的主要参考对象 [1]

### 7.3 我们的方法

- baseline 之后加一个轻量 reranking 模块
- 用 CNN 学习 similarity function
- 目标是在不明显增加开销的情况下提升排序质量

---

## 8. 实验设计

### 8.1 核心对比

至少做这两个：

1. **Baseline**
   - embedding + cosine similarity

2. **Ours**
  - baseline retrieval + CNN learned reranking

### 8.2 可选对比

如果时间允许，可以继续做：

- 不同 backbone：
  - CLIP
  - ResNet

- 不同 rerank 模块：
  - cosine（无精排）
  - CNN rerank（主方法）

### 8.3 评价指标

建议使用：

- Recall@K
- Top-K Accuracy
- qualitative visualization

### 8.4 可视化展示

必须做的图：

- query image
- baseline top-k results
- reranked top-k results

这样最直观，也最适合课程项目展示。

---

## 9. 报告大纲

### 1. Introduction

- 图像相似推荐 / retrieval 的问题背景
- baseline 的局限：固定相似度函数太粗
- 引出 reranking 的必要性
- 说明我们的目标：用轻量 learned reranking 替代更重的 transformer reranking

### 2. Related Work

- embedding-based retrieval
- cosine similarity based ranking
- transformer-based reranking
- STIR 和 ICCV 2021 Reranking Transformer 两篇工作 [1][2]

### 3. Method

- baseline pipeline
- Top-K retrieval
- pairwise reranking
- learned similarity function
- 我们的 CNN pairwise 设计（6 通道拼接输入）

### 4. Experiments

- dataset
- evaluation metrics
- baseline vs ours
- 可选扩展实验

### 5. Results and Discussion

- quantitative results
- qualitative cases
- 为什么 learned reranking 比 cosine 更好
- 为什么轻量方法比 transformer 更适合课程项目和低资源环境

### 6. Conclusion

- 总结方法
- 总结效果
- 提出后续可扩展方向（例如更强 reranker、multimodal rerank 等）

---

## 10. 最终一句话版本

> 我们先复现一个基于 embedding + cosine similarity 的图像检索系统作为 baseline，再在 Top-K 候选上加入轻量 CNN reranking：让 CNN 直接比较 `(query, candidate)` 图像对并输出相似度分数，用可学习的 similarity function 修正粗排结果，从而在较低计算开销下提升排序质量。

---

## 参考链接

[1]: https://arxiv.org/abs/2304.13393 "[2304.13393] STIR: Siamese Transformer for Image Retrieval Postprocessing"

[2]: https://openaccess.thecvf.com/content/ICCV2021/html/Tan_Instance-Level_Image_Retrieval_Using_Reranking_Transformers_ICCV_2021_paper.html "ICCV 2021 Open Access Repository"

[3]: https://github.com/tercasaskova311/Top-k-Image-Retrieval-Image-recognition-/tree/main "Top-k Image Retrieval baseline repository"
