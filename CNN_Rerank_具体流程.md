# CNN Rerank 具体流程说明

## 1. 整体逻辑：两阶段系统

系统分为两个阶段：

1. **粗排（Baseline Retrieval）**
2. **精排（CNN Reranking）**

粗排负责“快速筛候选”，精排负责“细粒度纠错排序”。

---

## 2. 第一步：粗排（Baseline）

Baseline 的流程：

- 输入 query image
- 用预训练 encoder 提取 query embedding
- 与 gallery embedding 做 cosine similarity
- 返回 Top-K 候选

作用：

- 从大规模图库中快速筛出“看起来较相关”的候选图

局限：

- cosine similarity 是固定公式
- 无法学习“什么样才是真正更像”
- 容易在细粒度排序上出错

---

## 3. 第二步：精排（CNN Rerank）

CNN rerank 不直接相信 baseline 的原始顺序，而是在 Top-K 候选内部再比较一次。

目标：

- 对每个 `(query, candidate)` 输出更细粒度的相似度分数
- 用新分数重排 Top-K

这一步本质是 **pairwise scoring**。

---

## 4. CNN Rerank 的输入与输出

### 输入

- query image
- candidate image

### 输出

- 一个实数分数 `score(query, candidate)`

分数解释示例：

- 0.95：非常像
- 0.72：比较像
- 0.20：不太像

该分数用于最终重排序。

---

## 5. 两图如何送入 CNN

采用“通道拼接”的直观方案：

- 单张 RGB 图：3 通道
- query 与 candidate 在通道维拼接后：6 通道

即 CNN 输入张量可表示为：

```text
X_pair = concat(query_rgb, candidate_rgb)  # C=6
```

然后 CNN 自动学习判别线索：

- 形状/轮廓是否一致
- 局部纹理是否一致
- 颜色差异是否应被强调或忽略
- 哪些差异是噪声，哪些差异反映类别/语义差异

---

## 6. 在线推理流程（Inference）

以 Top-20 为例：

1. Baseline 返回 Top-20
2. 构造 20 个图片对：
   - `(query, candidate1)`
   - `(query, candidate2)`
   - ...
   - `(query, candidate20)`
3. 每对拼接成 6 通道输入 CNN
4. 得到 20 个相似度分数
5. 按分数降序重新排序
6. 输出 reranked Top-K

可写成：

```text
score_i = CNN(concat(query, candidate_i))
final_rank = sort_by_score_desc(score_i)
```

---

## 7. 训练流程（Training）

### 训练样本形式

每个样本是三元组：

```text
(query, candidate, label)
```

- `label = 1`：相似（正样本）
- `label = 0`：不相似（负样本）

### 样本构造原则

如果数据集有类别标签：

- 同类图像对作为正样本
- 异类图像对作为负样本

### 训练目标

让 CNN 学会：

- 正样本输出高分
- 负样本输出低分

可将输出映射到 0~1 概率，用二分类损失训练。

---

## 8. 一个可落地的轻量 CNN 结构

最简实现（课程项目友好）：

- 输入：6 通道图像
- 若干层 Conv + ReLU + Pooling
- Flatten
- 1~2 层全连接
- 输出 1 个 similarity score

重点不是堆很深网络，而是完成“对图像对打分”的任务。

---

## 9. 为什么 CNN Rerank 合理

- Baseline 已经完成了大规模检索，精排只处理 Top-K，成本可控
- 小 CNN 就能体现 learned similarity 的核心价值
- 相比 Transformer rerank：
  - 实现更简单
  - 计算更轻
  - 更适合课程项目与低算力场景

项目故事线：

- **Baseline** 负责“快速粗筛”
- **CNN rerank** 负责“轻量精排”

---

## 10. 核心创新点（定位清晰版）

1. **用 CNN 做轻量 reranking**（而非 Transformer）
2. **学习 similarity function**（而非固定 cosine）
3. **做效果-开销 trade-off**（提升排序同时控制资源消耗）

---

## 11. 一句话总结

> baseline 先找出一批看起来像的候选图，再由小 CNN 对每个 `(query, candidate)` 进行相似度打分并重排，从而用可学习的比较规则修正粗排结果。
