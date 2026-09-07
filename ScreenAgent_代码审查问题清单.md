# ScreenAgent 代码审查问题清单

> 本清单基于对 `train.py`、`template.py`、`train_dataset.py`、`base_dataset.py`、
> `eval_dataset.py`、`evaluator.py`、`model_utils.py`、`inference.py` 八份文件的逐一审查整理而成。
> 文件的实际相对路径依据代码中的 import 语句推断（如 `from .template import ...`、
> `from data import HybridDataset`、`from models import find_lora_target_modules`），
> 实际仓库目录结构如与推断不符，请以仓库实际路径为准，本文档中仅使用文件名定位。
>
> **范围说明**：本次审查发现的、与"多任务（text2point/text2bbox/point2text/bbox2text）
> 混合训练能力尚未真正激活"相关的问题，统一归入文末【暂缓处理】章节，仅作记录，
> 不在本次修复范围内，请勿做增量设计。

---

## 严重程度说明

| 级别 | 含义 |
|---|---|
| 🔴 Critical | 会导致模型/系统产出**静默错误的结果**，且不会抛出任何异常提示，最难被发现，优先级最高 |
| 🟠 High | 在特定条件下会导致**程序崩溃**或**结果被系统性污染**，条件明确、可复现 |
| 🟡 Medium | 影响**性能、鲁棒性或问题排查效率**，不必然导致错误结果，但会放大其他问题的排查难度 |
| ⚪ Low | 代码质量/可维护性问题，暂不影响功能正确性 |

---

## 问题总览表

| 编号 | 严重程度 | 一句话描述 | 涉及文件 |
|---|---|---|---|
| C1 | 🔴 Critical | 坐标系统三处不一致，评估/推理结果可能系统性错误 | train_dataset.py, evaluator.py, inference.py |
| C2 | 🔴 Critical | LoRA 目标模块默认排除逻辑被有效绕过，`lm_head` 可能被意外纳入训练 | model_utils.py, train.py |
| H1 | 🟠 High | 轮询采样模式下 `train_ratio` 参数被静默忽略 | base_dataset.py |
| H2 | 🟠 High | `pixel_values` 拼接逻辑在多样本+不同分辨率场景下可能报错 | base_dataset.py |
| H3 | 🟠 High | `attention_mask` 被删除后从未重新生成 | template.py, base_dataset.py |
| M1 | 🟡 Medium | `sdpa` 模式下加速后端被意外关闭，退化为最慢实现 | train.py |
| M2 | 🟡 Medium | 评估阶段裸 `except` 吞掉所有异常，掩盖代码bug | evaluator.py |
| M3 | 🟡 Medium | 推理脚本 `smart_resize` 缺少输入合法性校验 | inference.py |
| M4 | 🟡 Medium | 推理脚本坐标解析鲁棒性弱于评估脚本 | inference.py |
| M5 | 🟡 Medium | 核心逻辑（resize、模板、坐标转换）三处重复实现 | train_dataset.py, evaluator.py, inference.py |
| L1 | ⚪ Low | 无用的量化配置 import | train.py |
| L2 | ⚪ Low | 工具函数定义但未被实际调用 | model_utils.py |
| L3 | ⚪ Low | 部分随机决策未绑定可复现的随机种子 | train_dataset.py |

---

## 🔴 C1：坐标系统三处不一致，评估/推理结果可能系统性错误

### 位置

1. `train_dataset.py` → `GroundingTrainDataset._build_single_turn`（point 任务分支）
2. `train_dataset.py` → `GroundingTrainDataset._build_multi_turn`（point 任务分支，调用 `_convert_point_for_qwen`）
3. `evaluator.py` → `evaluate_screenspot` 内部，`convert_point_from_qwen_format` 的调用处
4. `inference.py` → `ScreenAgentInference.predict` 内部，`_convert_point_to_original` 的调用处

### 现象

同一个 point 定位任务，在训练、评估、推理三个环节，对"模型输出的坐标应该是什么坐标系"这件事，做出了**互相矛盾**的假设：

- `_build_single_turn`（默认单轮训练，`num_turn=1` 是 `train.py` 的默认值）产出的标准答案坐标是**归一化坐标**：
  ```python
  if self.xy_int:
      answer_xy = [int(x * 1000) for x in answer_xy]   # 0~1000 整数
  else:
      answer_xy = [round(x, 2) for x in answer_xy]      # 0~1 小数
  ```
- `_build_multi_turn` 的 point 任务分支，当模型是 Qwen2.5-VL 时，却调用 `_convert_point_for_qwen`，产出的是**图片经过智能 resize 后的绝对像素坐标**，与单轮完全是两套不同的数值体系。
- `evaluator.py` 中，只要检测到 `'Qwen2.5-VL' in args.model_id`，**无条件**调用 `convert_point_from_qwen_format`，把模型输出一律当作"resize 后绝对坐标"去反解换算，不区分这条评估数据对应的模型实际是按单轮（归一化坐标）还是多轮（绝对坐标）方式训练出来的。
- `inference.py` 的 `_convert_point_to_original` 同样无条件按"resize 后绝对坐标"处理模型输出。

### 原因分析

`train.py` 默认 `--num_turn 1`，也就是说**默认训练配置下，模型学到的输出格式是归一化坐标**，而不是 resize 后绝对坐标。但 `evaluator.py` 和 `inference.py` 都是按"resize 后绝对坐标"的假设写的换算逻辑。

以一个具体数值算一遍偏差：假设原图 1920×1080，Qwen2.5-VL 智能 resize 后变为约 980×560（具体数值取决于 `min_pixels`/`max_pixels` 设置）。如果模型是按单轮方式训练的，输出"屏幕正中心"会是归一化坐标 `[500, 500]`（0~1000 整数格式）。但评估/推理代码会把这个 `[500, 500]` **误当作"980×560 坐标系里的一个点"**，按比例换算为：

```
x_orig = 500 * (1920/980) ≈ 980
y_orig = 500 * (1080/560) ≈ 964
```

而真实的原图正中心应为 `(960, 540)`。换算结果偏离明显，会导致本该判定为"预测正确"的样本，被误判为"未落在目标框内"，从而使 `evaluate_screenspot` 算出的 accuracy 被系统性拉低；`inference.py` 的最终输出坐标也会明显偏移，导致点击定位失准。

**这是一个不会抛出任何异常的静默错误**——代码能正常跑完、能输出一个"看起来合理"的坐标数字，但这个数字在数值上是错的，只有对照实际效果才能发现。

### 修改建议

需要先明确一个业务决策：**point 任务的标准输出坐标系，最终要统一成哪一种？** 建议二选一：

**方案 A（推荐，改动更小）：统一采用归一化坐标（`xy_int` 控制的 0~1000 整数或 0~1 小数）**
- 修改 `_build_multi_turn` 中 point 任务分支，去掉 `_convert_point_for_qwen` 这条特殊路径，改为与 `_build_single_turn` 一致的归一化坐标处理方式。
- 修改 `evaluator.py`：去掉无条件调用 `convert_point_from_qwen_format` 的逻辑，改为根据训练时使用的坐标格式标记，选择"按归一化坐标反解（乘回原图宽高）"还是"按 resize 绝对坐标反解"。
- 修改 `inference.py` 的 `_convert_point_to_original` / `predict`，同步改为按归一化坐标反解。

**方案 B：统一采用 resize 后绝对坐标**
- 修改 `_build_single_turn`，改为调用类似 `_convert_point_for_qwen` 的逻辑，使单轮 point 任务的标准答案也变为 resize 后绝对坐标。
- `evaluator.py` / `inference.py` 保持现状不变。

无论选择哪种方案，都建议：
- 在训练产出的 checkpoint 目录（如 `args.log_dir/args.json`）或模型 config 中，**显式记录本次训练实际使用的坐标格式**（例如新增一个字段 `coord_format: "normalized_int1000" | "normalized_float" | "qwen_absolute"`）。
- `evaluator.py` 和 `inference.py` 加载模型时读取这个标记，据此选择正确的反解方式，而不是硬编码判断 `'Qwen2.5-VL' in model_id`。这样即便未来坐标格式再调整，评估/推理端也能自动适配，不需要再次人工同步三处代码。

### 验证建议

修复后，建议构造一个已知坐标的合成测试用例（比如手工标注一张图片上某元素的中心点），分别走一遍训练时的编码流程和评估/推理时的解码流程，确认往返转换后坐标数值一致（允许取整误差）。

---

## 🔴 C2：LoRA 目标模块默认排除逻辑被有效绕过

### 位置

- `model_utils.py` → `find_lora_target_modules` 函数定义
- `train.py` → 调用 `find_lora_target_modules` 处：
  ```python
  exclude_modules = ["visual"] if not args.tune_visual_encoder else []
  target_modules = find_lora_target_modules(model, exclude_keywords=exclude_modules)
  ```

### 现象

`find_lora_target_modules` 的文档字符串声明"默认排除视觉编码器和 `lm_head`"：

```python
if exclude_keywords is None:
    exclude_keywords = [
        "visual", "vision_model", "img_projection", "lm_head",
    ]
```

但这个默认列表**只有在调用方完全不传 `exclude_keywords` 参数（即保持 `None`）时才会生效**。而 `train.py` 的实际调用永远显式传入了 `exclude_keywords=exclude_modules`（不管 `tune_visual_encoder` 是否为真，传入的都是一个具体列表，不是 `None`），导致函数内置的默认排除列表整体被跳过。

代入两种实际配置验证：

- `tune_visual_encoder=False`（默认）：`exclude_modules = ["visual"]`，**`lm_head` 未被排除**，会被当作 LoRA 目标模块之一，与文档字符串描述的"默认排除 lm_head"意图不符。
- `tune_visual_encoder=True`：`exclude_modules = []`，**什么都不排除**，`lm_head` 和视觉编码器内部的 Linear 层都会被纳入 LoRA。

### 原因分析

函数设计的意图是"提供一个开箱即用的默认排除清单，调用方可按需覆盖"，但 `train.py` 的调用方式（无条件显式传参）使这个"默认清单"从未真正生效过。这是函数默认参数设计与实际调用方式脱节导致的逻辑漏洞。

对 `lm_head` 做 LoRA 微调，相当于额外调整了"整个词表的输出概率分布偏好"，这和"只调整 attention/FFN 内部表示、保持输出层不变"是两种风险和收益完全不同的策略，是否要对 `lm_head` 做 LoRA，应该是一个需要明确决策、而不是被默认参数设计的疏漏意外触发的事情。

### 修改建议

需要先确认业务意图：**`lm_head` 到底应不应该参与 LoRA 微调？**

**若不应该参与（推荐，与原文档字符串意图一致）：**
```python
# train.py
exclude_modules = ["visual", "lm_head"] if not args.tune_visual_encoder else ["lm_head"]
target_modules = find_lora_target_modules(model, exclude_keywords=exclude_modules)
```

**若希望函数级别更健壮，不依赖调用方记得手动加 `lm_head`：**
修改 `find_lora_target_modules` 内部逻辑，把"内置默认排除词"和"调用方额外传入的排除词"合并，而非互斥替代：
```python
def find_lora_target_modules(model, exclude_keywords=None, num_modules=-1, verbose=True):
    default_exclude = ["visual", "vision_model", "img_projection", "lm_head"]
    if exclude_keywords is None:
        exclude_keywords = default_exclude
    else:
        exclude_keywords = list(set(default_exclude) | set(exclude_keywords))
    ...
```
这样无论调用方是否显式传参，`lm_head` 等内置默认排除项都始终生效，调用方传入的参数只是"额外增加"排除项，不会意外撤销默认保护。

**若确实希望 `lm_head` 参与训练：** 保持现状不改代码，但需要更新函数文档字符串，去掉"默认排除 lm_head"这一条不实描述，避免误导后续维护者。

### 验证建议

修复后，在 `model.print_trainable_parameters()` 打印结果中，或直接遍历 `target_modules` 列表，确认 `lm_head` 是否按预期出现/不出现。

---

## 🟠 H1：轮询采样模式下 `train_ratio` 参数被静默忽略

### 位置

`base_dataset.py` → `HybridDataset.__getitem__`，`elif self.random_sample and self.record_sample:` 分支

### 现象

```python
elif self.random_sample and self.record_sample:
    # 轮询采样：保证每个数据集都被均匀采样
    ds_idx = self.current_dataset_idx
    ...
    self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.datasets)
    return dataset[sample_idx]
```

该分支完全没有使用 `self.sample_rates`（对应命令行参数 `--train_ratio`），只是严格按 `current_dataset_idx` 依次轮询访问每个子数据集，各数据集被访问的频率**完全相等**，与用户通过 `--train_ratio` 设定的比例无关。

### 原因分析

用户如果同时设置了 `--random_sample --record_sample --train_ratio "3,1,1,1"`，期望某个数据集被更多地采样，但实际运行结果是四个数据集被均匀轮询访问，`train_ratio` 参数在这种模式组合下是"哑"参数，不会有任何提示告知用户这一点，容易造成"配置了但没生效、且不自知"的困惑。

### 修改建议

需要先确认业务意图：

- **若"轮询模式本意就是完全均匀采样，不考虑比例"是预期行为**：应在参数帮助文本（`add_argument` 的 `help=`）或运行时日志中明确提示，例如在 `HybridDataset.__init__` 中检测到 `record_sample=True` 且 `sample_rates` 并非均匀分布时，打印一条 warning：
  ```python
  if self.record_sample and not np.allclose(self.sample_rates, self.sample_rates[0]):
      print("[警告] record_sample=True 时，train_ratio 设置的采样比例不会生效，各数据集将被均匀轮询访问")
  ```
- **若轮询模式也应该尊重比例**：需要重新设计为"加权轮询"（weighted round-robin），例如按 `sample_rates` 归一化后的比例，动态调整每个数据集在一轮访问序列中出现的次数，同时保留"记录已访问样本、轮空重置"这个去重机制。

### 验证建议

修复后，建议写一个简单的统计脚本，多次调用 `HybridDataset.__getitem__`，统计各数据集实际被访问的次数分布，与预期（均匀或按比例）做对比验证。

---

## 🟠 H2：`pixel_values` 拼接逻辑在多样本+不同分辨率场景下可能报错

### 位置

`base_dataset.py` → `collate_fn`：

```python
if len(pixel_values[0].shape) == 2:
    pixel_values = torch.stack(pixel_values, dim=0)
else:
    pixel_values = torch.cat(pixel_values, dim=0)
```

### 现象

Qwen2.5-VL 支持动态分辨率，不同图片编码后的 `pixel_values` 在 `num_patches` 这一维度上很可能长度不同。`torch.stack` 要求所有被堆叠的张量形状**完全一致**；如果 `batch_size > 1` 且这一批样本恰好包含分辨率差异较大的图片（导致 `pixel_values` 形状分别是如 `[350, hidden_dim]` 和 `[420, hidden_dim]`），这里判断走到 `torch.stack` 分支会直接抛出 shape mismatch 运行时错误。

### 原因分析

`train.py` 默认 `--batch_size 1`，此时 `pixel_values` 列表只有一个元素，`torch.stack([单个tensor], dim=0)` 相当于简单加一个 batch 维度，不会触发报错，问题被"默认配置恰好掩盖"。一旦有人将 `batch_size` 调大，且 DataLoader 采样到不同分辨率的图片组合在同一个 batch，就会触发报错。

### 修改建议

- 明确记录当前实现只在 `batch_size=1` 下保证正确，建议在 `train.py` 解析完 `args.batch_size` 后加入一条校验/警告：
  ```python
  if args.batch_size > 1:
      print("[警告] 当前 collate_fn 对 batch_size>1 且图片分辨率不一致的情况未做充分验证，可能报错")
  ```
- 若确实需要支持 `batch_size > 1`：统一使用 `torch.cat` 而非条件判断 `stack`/`cat`，因为对于形状可能不一致的 `num_patches` 维度，天然应该走拼接而非堆叠，并确保 `image_grid_thw`（即 `image_sizes`）正确携带了每张图各自的 patch 划分信息，供模型内部正确切分。

### 验证建议

构造一个单元测试：手动构造两张分辨率差异较大的图片进入同一个 batch（设置 `batch_size=2`），跑一遍 `collate_fn`，确认是否复现报错；修复后重新验证是否能正常通过。

---

## 🟠 H3：`attention_mask` 被删除后从未重新生成

### 位置

- `template.py` → `add_answer_to_batch` 和 `add_multiturn_answer`，均有：
  ```python
  if 'attention_mask' in batch:
      del batch['attention_mask']
  ```
- `base_dataset.py` → `collate_fn`，从头到尾未见任何重新生成 `attention_mask` 的逻辑，最终 `result` 字典中也不包含 `attention_mask` 这个 key。

### 现象

`add_answer_to_batch`/`add_multiturn_answer` 拼接答案后，原有的（对应"仅提示词"部分长度的）`attention_mask` 已经和新的、更长的 `input_ids` 长度对不上，因此被主动删除，意图是"交给后续步骤重新生成"。但通读 `collate_fn` 全部逻辑，并未找到重新生成它的代码，意味着最终喂给模型的 batch 中**不包含 `attention_mask`**。

### 原因分析

如果模型的 `forward`/`generate` 方法在 `attention_mask=None` 时默认将所有位置视为"可见"（这是许多因果语言模型的常见默认行为），那么 batch 内经过 padding 补齐的部分（`pad_token_id` 对应的位置）可能被错误地纳入 attention 计算，污染真实 token 的表示。这个问题的严重性取决于所用 `transformers` 版本对应模型类的具体默认实现，需要针对项目实际使用的版本做确认。

### 修改建议

在 `collate_fn` 中，基于 padding 后的 `input_ids` 和 `pad_token_id`，显式重新生成 `attention_mask`：

```python
attention_mask = (input_ids != pad_token_id).long()
...
result = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,   # 新增
    'labels': labels,
    ...
}
```

同时需要确认 `trainer/trainer.py` 中调用 `model_engine(...)` 的地方，是否已经把 `attention_mask` 传给模型（如果训练循环里本来就没传，这条修复需要同步在训练循环调用处补上）。

### 验证建议

修复前后，分别打印/检查同一个 batch 送入模型前的完整 kwargs，确认 `attention_mask` 是否存在、形状是否与 `input_ids` 一致、padding 位置是否被正确标记为 0。若有条件，可以做一次简单的对照实验，比较修复前后模型在验证集上的 loss/accuracy 是否有变化，以判断该问题在实践中影响的严重程度。

---

## 🟡 M1：`sdpa` 模式下加速后端被意外关闭

### 位置

`train.py`：

```python
if args.attn_imple in ["eager", "sdpa"]:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
```

### 现象

当用户选择 `--attn_imple sdpa`（这是 `train.py` 的默认值）时，这两行代码依然会被执行，把 PyTorch 底层的高效 attention 加速后端全部关闭。但这两个开关恰恰是 `sdpa`（`scaled_dot_product_attention`）内部用来自动选择加速路径的开关，全部关闭会导致 `sdpa` 退化为最基础、最慢的实现路径。

### 原因分析

这段逻辑对 `eager` 和 `sdpa` 一视同仁地关闭加速后端，符合 `eager` 的语义（既然选择朴素实现，就该完全走朴素路径），但不符合选择 `sdpa` 的初衷（`sdpa` 通常正是为了"兼顾通用性和性能"而被选用，选它却又把它的加速能力关掉，逻辑上自相矛盾）。

### 修改建议

将条件收窄，只在 `eager` 模式下关闭这两个开关：

```python
if args.attn_imple == "eager":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
```

### 验证建议

修改后，建议在相同硬件、相同数据、相同步数的条件下，对比修改前后 `sdpa` 模式的训练吞吐量（每秒处理的样本数/耗时），确认收益，并确认训练结果（loss 曲线）没有异常变化。

---

## 🟡 M2：评估阶段裸 `except` 吞掉所有异常

### 位置

`evaluator.py` → `evaluate_screenspot`，计算 acc 的部分：

```python
try:
    pred_point = ast.literal_eval(pred)
    step_result['pred_point'] = pred_point
    if point_in_bbox(pred_point, gt_bbox):
        step_result["acc"] = 1
    else:
        step_result["acc"] = 0
except:
    step_result["acc"] = 0
```

### 现象

这里的裸 `except:` 会捕获**所有类型**的异常，包括本不该被静默吞掉的代码逻辑错误（比如标注数据字段缺失、`gt_bbox` 格式异常等），一律归类为"模型预测失败，acc=0"，导致代码本身的 bug 有可能被长期掩盖、误判为"模型能力不足"。

### 原因分析

评估流程中"模型输出格式不规整"和"代码/数据本身有问题"是两类完全不同性质的问题，前者应该被容忍并计入失败样本，后者应该被开发者及时发现并修复。当前实现无法区分这两种情况。

### 修改建议

改为捕获具体的、预期内的异常类型，对预期外的异常应重新抛出或至少记录详细日志：

```python
try:
    pred_point = ast.literal_eval(pred)
    step_result['pred_point'] = pred_point
    step_result["acc"] = 1 if point_in_bbox(pred_point, gt_bbox) else 0
except (ValueError, SyntaxError, TypeError) as e:
    logging.warning(f"预测结果解析失败 anno_id={output.get('anno_id')}: {pred!r}, error={e}")
    step_result["acc"] = 0
```

如遇到不在预期范围内的异常类型（例如 `KeyError`），不应该被静默吃掉，建议保留 traceback 输出或直接抛出，便于定位是否是代码层面的问题。

### 验证建议

修复后，人为构造几种边界输入（如 `gt_bbox` 缺失字段、`pred` 是合法但非二元组的字面量），确认异常处理行为符合预期：预期内的格式错误被正确记为 acc=0 并留有日志，预期外的错误能被开发者感知到。

---

## 🟡 M3：推理脚本 `smart_resize` 缺少输入合法性校验

### 位置

`inference.py` → `ScreenAgentInference._smart_resize`，对比 `train_dataset.py` → `_smart_resize` 和 `evaluator.py` → `smart_resize`

### 现象

`train_dataset.py` 和 `evaluator.py` 中的 `smart_resize` 均包含以下两处校验：

```python
if height < factor or width < factor:
    raise ValueError(f"图片太小: {height}x{width}")
if max(height, width) / min(height, width) > 200:
    raise ValueError(f"长宽比太极端: {height}x{width}")
```

但 `inference.py` 中的同名函数**没有**这两处校验。

### 原因分析

如果用户在推理阶段上传了异常尺寸的截图（过小或长宽比极端），训练/评估阶段的代码会主动抛出明确异常提示；而部署给最终用户使用的推理脚本会直接跳过校验、继续往后计算，可能得到一个数值上"能跑但没有实际意义"的坐标结果，且不会有任何报错提示开发者或用户"这张图片有问题"。

### 修改建议

在 `inference.py` 的 `_smart_resize` 中补齐相同的两处校验逻辑，与 `train_dataset.py`/`evaluator.py` 保持一致（理想情况下应提取为共享函数，见 M5 建议）。

### 验证建议

构造一张边长小于 28 像素的极小图片、以及一张长宽比超过 200:1 的极端图片，分别调用 `ScreenAgentInference.predict`，确认能够得到明确的报错提示，而不是静默产出一个无意义的坐标。

---

## 🟡 M4：推理脚本坐标解析鲁棒性弱于评估脚本

### 位置

`inference.py` → `ScreenAgentInference.predict`：

```python
try:
    pred_point = ast.literal_eval(output_text.strip())
    ...
except Exception as e:
    print(f"解析坐标失败: {e}")
    return None, None
```

对比 `evaluator.py` → `extract_coordinates` + 后续解析逻辑。

### 现象

`evaluator.py` 在做 `ast.literal_eval` 之前，先用正则表达式尝试从生成文本中"抠出"符合 `[x, y]` 格式的子串，能够容忍模型输出中夹杂自然语言修饰（如"点击位置是[512, 300]"）。而 `inference.py` 直接对整段生成文本做 `ast.literal_eval`，一旦文本不是"恰好合法的 Python 字面量"，会直接解析失败返回 `(None, None)`，容错能力明显弱于评估脚本。

### 原因分析

面向最终用户的推理接口理应具备不低于内部评估脚本的鲁棒性，当前实现在这一点上出现了"生产环境代码反而比内部工具脚本更脆弱"的反常情况，大概率是两份代码各自独立演化、未同步的结果。

### 修改建议

复用（或迁移引入）`evaluator.py` 中 `extract_coordinates` 的正则提取逻辑，在 `ast.literal_eval` 之前先尝试从文本中提取符合 `[x, y]` 格式的子串：

```python
import re

def extract_coordinates(text):
    match = re.search(r'\[\s*[\d\.]+\s*,\s*[\d\.]+\s*\]', text)
    return match.group(0) if match else None

# predict() 内部：
coord_str = extract_coordinates(output_text)
pred_point = ast.literal_eval(coord_str if coord_str else output_text.strip())
```

（若采纳 M5 的共享模块建议，这个函数应直接从共享模块导入，而不是在 `inference.py` 内再复制一份。）

### 验证建议

构造几个模型输出样例，包括"纯坐标"和"坐标前后带文字修饰"两种情况，确认 `predict` 方法在两种情况下都能正确解析出坐标。

---

## 🟡 M5：核心逻辑三处重复实现，维护风险高

### 位置

以下三组逻辑，分别在 `train_dataset.py`、`evaluator.py`、`inference.py` 中各自独立实现了一份：

1. `smart_resize` / `_smart_resize`（图像智能 resize 计算）
2. `CHAT_TEMPLATE`（Jinja2 聊天模板字符串，`train.py` 和 `inference.py` 中各硬编码一份完全相同的长字符串）
3. 坐标转换逻辑：`_convert_point_for_qwen`（train_dataset.py）、`convert_point_from_qwen_format`（evaluator.py）、`_convert_point_to_original`（inference.py），三者逻辑等价但各自独立实现

### 现象

同一段逻辑分散在三个文件中各写一份，修改任何一处时容易遗漏同步其余两处。M3 中发现的"推理脚本缺少输入校验"，正是这种重复导致的直接后果之一。

### 原因分析

这种重复大概率是不同时间点开发、缺乏统一的共享工具模块所致。`inference.py` 不依赖训练框架（不 import deepspeed/wandb 等训练专属重型依赖）本身是合理的部署考量，但当前的实现方式（直接复制粘贴、各自维护）以增加长期维护成本和引入不一致 bug 为代价，换取了这种"自包含"。

### 修改建议

提取一个独立的共享工具模块（建议命名如 `common/qwen_vl_utils.py` 或 `utils/coord_utils.py`），集中存放：
- `smart_resize` 函数
- 坐标转换函数（training/qwen 坐标 ↔ 原图坐标）
- `CHAT_TEMPLATE` 常量
- （建议一并纳入）`extract_coordinates` 正则提取函数（见 M4）

该模块应**只依赖 torch、PIL 等推理必需的基础库**，不引入 deepspeed、wandb、peft 等训练专属重型依赖，这样 `inference.py` 仍可以保持"轻量部署、不依赖完整训练框架"的特性，同时 `train_dataset.py`、`evaluator.py`、`inference.py` 三处统一从这个模块 import，杜绝重复代码。

### 验证建议

重构完成后，对三个文件分别跑一次现有的（或补充的）单元测试，确认重构没有改变任何实际计算结果，仅仅是消除了代码重复。

---

## ⚪ L1：无用的量化配置 import

### 位置

`train.py` 顶部：

```python
from transformers import AutoProcessor, BitsAndBytesConfig
```

### 现象

`BitsAndBytesConfig` 被导入，但全文件未见任何实际使用（没有被实例化，也没有被传给 `from_pretrained`）。

### 原因分析

可能是量化训练（QLoRA）功能的历史遗留代码，或是预留但尚未完成开发的半成品功能。

### 修改建议

- 若当前无量化训练计划：直接删除这一处无用 import。
- 若确实计划支持 QLoRA：需要补充完整的量化配置解析（新增命令行参数）和模型加载逻辑（`from_pretrained` 时传入 `quantization_config=...`）。**这属于新功能开发，不在本次"修复已有问题"范围内，建议单独立项处理，本次仅需先去除死代码或添加 TODO 注释说明。**

---

## ⚪ L2：工具函数定义但未被实际调用

### 位置

`model_utils.py` → `print_trainable_parameters`、`freeze_module`、`unfreeze_module`

### 现象

这三个函数在目前审查过的项目代码范围内均未被实际调用：
- `train.py` 中打印可训练参数用的是 PEFT 库自带的 `model.print_trainable_parameters()` 方法，而非 `model_utils.py` 里自己写的同名函数。
- `train.py` 中冻结视觉编码器用的是手写循环 `for p in model.base_model.model.visual.parameters(): p.requires_grad = False`，而非调用 `freeze_module(...)`。

### 原因分析

大概率是后续重构时补充的"通用化尝试"，但没有回头替换掉已有的调用点，导致功能重复、两套实现并存。

### 修改建议

- 若确认这些工具函数当前没有实际使用场景：建议清理移除，减少代码维护负担。
- 若希望保留作为工具库对外接口供其他模块复用：建议回填替换 `train.py` 中对应的手写逻辑，改为实际调用这些工具函数，避免重复维护两份等价代码。

---

## ⚪ L3：部分随机决策未绑定可复现的随机种子

### 位置

`train_dataset.py` → `GroundingTrainDataset._get_sample`：

```python
task_type = np.random.choice(len(self.sample_prob), p=self.sample_prob)   # 未设种子
...
if self.num_turn == 1:
    random.seed(idx)                                    # 设了种子，可复现
    element_idx = random.randint(0, len(elements) - 1)
else:
    selected_elements = random.choices(elements, k=num_elements)   # 未设种子
```

### 现象

单轮模式下的元素选择显式调用了 `random.seed(idx)`，保证"同一个 idx，每次调用结果完全一致（可复现）"；但 `task_type` 的选择、以及多轮模式下的元素选择，都没有绑定与 `idx` 相关的随机种子，导致同一个 `idx` 在不同次调用之间，返回的具体训练样本内容不完全可复现。

### 原因分析

不清楚这种不一致是有意为之（比如认为"任务类型的随机性不需要强可复现，元素选择更需要可复现以便调试"），还是遗漏。

### 修改建议

- 若训练数据的完全可复现性有实际调试需求：建议统一使用基于 `idx`（必要时结合 epoch/step）派生的随机种子，控制所有随机决策点（任务类型选择、单轮/多轮元素选择），确保给定相同 `idx` 时行为完全确定。
- 若当前的不确定性是有意为之（增加同一 `idx` 被重复访问时的数据多样性）：建议在代码注释中明确说明这一设计取舍，避免后续维护者误判为 bug 并"修复"掉这个有意为之的随机性。

---

## 【暂缓处理】与多任务相关的已知问题（仅记录，不要求本次修改）

以下问题均与"text2point / text2bbox / point2text / bbox2text 四类任务混合训练"相关。
根据要求，多任务能力本次**暂不做激活或增量设计**，以下内容仅作记录存档，供未来需要时参考：

1. `train.py` 的 `argparse` 只注册了 `--text2point`、`--text2bbox` 两个参数，未注册
   `--point2text`、`--bbox2text`。`train_dataset.py` 中 `args_dict.get('point2text', 0)`、
   `args_dict.get('bbox2text', 0)` 因此永远只能取到默认值 0，导致默认配置下
   `sample_prob` 实际归一化为 `[1.0, 0.0, 0.0, 0.0]`，四类任务混合训练在当前对外暴露的
   命令行接口下，实质上从未被真正激活过，仅有 text2point 一种任务在运行。
2. `template.py` 的 `build_eval_prompt` 函数不接受 `sample_type` 参数，`eval_dataset.py`
   中两处评估数据集调用它时，构造出的评估问题格式固定为 text2point 风格，不覆盖
   另外三类任务的评估问题构造。
3. `evaluator.py` 的 `extract_coordinates` 正则表达式只支持二元组 `[x, y]` 格式，
   `point_in_bbox` 判分逻辑也只针对"点是否落在框内"设计，均不支持 bbox 格式
   （四元组）或文字生成类任务（point2text/bbox2text）的判分。
4. `train_dataset.py._get_sample` 中，多轮模式下元素选择使用
   `random.choices(elements, k=num_elements)`（有放回抽样），轮次之间的元素选择
   相互独立、无任何指代/排除/顺序关系，多轮对话训练目前本质上是"同一张图内
   若干个互不相关（甚至可能重复）的独立问答被物理拼接成一条长序列"，
   不构成真正需要"利用上下文"能力的训练信号。

---

## 文档使用说明（供 Claude Code 参考）

- 建议按 **C → H → M → L** 的顺序处理，C1、C2 属于会静默产出错误结果的问题，优先级最高。
- 每一项修改后，请参照对应条目末尾的【验证建议】做针对性验证，而不仅仅是让代码能跑通。
- 修改 C1 前，请先与项目负责人确认"point 任务坐标系统最终应统一成哪一种格式"这一业务决策，
  该决策会影响后续具体改动方式（本文档提供的方案 A / 方案 B 两条路径）。
- 【暂缓处理】章节中的问题**不需要修改代码**，如果在修复过程中顺带发现这些代码路径，
  请保持现状，仅在注释中标注"已知限制，详见问题清单暂缓处理章节"即可，不要顺手做增量开发。
