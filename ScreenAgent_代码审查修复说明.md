# ScreenAgent 代码审查修复说明

> 本文档对应 `ScreenAgent_代码审查问题清单.md` 中列出的 13 个问题（C1、C2、H1-H3、M1-M5、L1-L3），
> 逐条说明**做了什么修改、为什么这么改**，以及**修改前需要的业务决策是如何定的**。
> 【暂缓处理】章节（多任务混合训练相关的 4 个已知限制）本次未做任何代码改动，维持原样。

修复分两批完成：第一批（C1、C2、H1-H3）关乎"会不会产出静默错误结果 / 会不会崩溃"，优先级最高；
第二批（M1-M5、L1-L3）是性能、鲁棒性和代码质量问题，在第一批验证通过后进行。

---

## 第一批：Critical + High

### C1：point 任务坐标系统统一为「resize 后绝对坐标」

**问题**：训练时单轮（`_build_single_turn`）产出归一化坐标标签，多轮（`_build_multi_turn`，
Qwen2.5-VL 分支）产出 resize 后绝对像素坐标标签，两者互相矛盾；而 `evaluator.py`/`inference.py`
无条件按"绝对坐标"解析模型输出。更严重的是，`train_stage1.sh` 实际配置 `num_turn=30`，
但 `_get_sample` 是按"这张图恰好选出几个元素"决定走单轮还是多轮分支，与 `num_turn` 设置无关——
导致**同一次训练里，不同图片产出的标签坐标系统就不一致**。

**决策**：采用**方案 B——统一为 resize 后绝对坐标**（而不是改成归一化坐标）。
`evaluator.py`/`inference.py` 保持原有假设不变，只需修正训练侧的产出与之对齐。

**修改**：`data/train_dataset.py` 的 `_build_single_turn` 中，point 任务分支改为和
`_build_multi_turn` 完全一致的判断逻辑——`'Qwen2.5-VL' in self.model_id` 时调用
`_convert_point_for_qwen` 产出绝对坐标，否则才退回归一化坐标。这样无论一张图被分发到
单轮还是多轮构建路径，产出的坐标系统都相同，从根上解决"训练内部自相矛盾"的问题。

**为什么这么改**：改动面最小——`evaluator.py`/`inference.py` 已经是按绝对坐标写的，
改训练侧比同时改评估、推理两处风险更低；且这样一改，单轮/多轮分支从"两套坐标语义"
变成"同一套坐标语义、只是轮数不同"，顺带修掉了文档里没写出来的那个"同一次训练内部
坐标系统不一致"的深层问题。

---

### C2：LoRA 默认排除逻辑不再被绕过，`lm_head` 不参与训练

**问题**：`train.py` 调用 `find_lora_target_modules` 时永远显式传入 `exclude_keywords`
（无论是 `["visual"]` 还是 `[]`），导致函数内置的默认排除列表（含 `lm_head`）从未生效。

**决策**：`lm_head` **不参与** LoRA 微调（与函数文档字符串原本的意图一致）。

**修改**：
1. `models/model_utils.py`：`find_lora_target_modules` 改为"内置默认排除词 ∪ 调用方传入的
   排除词"的合并逻辑，而不是调用方一传参就整体替换默认值。
2. `trainer/train.py`：调用处加注释说明 `lm_head` 由函数内部保证排除。

**为什么这么改**：只修 `train.py` 的调用参数虽然能解决当前这一处问题，但不解决根因——
函数的参数设计本身就是"一传参就丢默认值"，未来任何新调用点都可能重蹈覆辙、且不会有任何
报错提示。把合并逻辑下沉到函数内部，才能让 `lm_head` 的排除**不依赖调用方是否记得**，
一劳永逸。

---

### H1：轮询采样模式下，明确提示 `train_ratio` 不生效

**问题**：`random_sample=True, record_sample=True`（轮询采样）时完全不看 `sample_rates`，
各数据集被严格均匀访问，`train_ratio` 是"哑参数"且没有任何提示。

**决策**：**保留"轮询模式=严格均匀"的现有行为**，不改造成加权轮询（改动小、不引入新的
采样逻辑设计风险），但在配置了非均匀比例时给出明确警告，避免用户"配置了却不自知"。

**修改**：`data/base_dataset.py` 的 `HybridDataset.__init__` 中，检测到
`random_sample and record_sample` 且 `sample_rates` 并非均匀分布时打印警告。

**为什么这么改**：轮询模式的本意就是"确保每个数据集都被完整、均匀地过一遍"，这和
"按比例采样"的诉求本身就有点冲突（想要不均匀比例的话应该用非轮询的随机采样模式）；
把两者硬融合成加权轮询需要重新设计采样状态机、且没有明确的业务需求驱动，投入产出比不高，
先加提示、把选择权交回用户更稳妥。

---

### H2：`pixel_values` 拼接统一使用 `torch.cat`

**问题**：`collate_fn` 用 `len(shape) == 2` 来决定 `stack` 还是 `cat`，`batch_size=1`
时因为列表只有一个元素而恰好不报错，一旦 `batch_size > 1` 且样本分辨率不同，
`torch.stack` 会因为形状不一致直接抛异常。

**修改**：去掉这个条件判断，统一使用 `torch.cat`。

**为什么这么改**：Qwen2.5-VL 的 `pixel_values` 本身是按 patch 拼接、不带独立 batch 维的
2D 张量（`[num_patches, hidden_dim]`），`image_grid_thw` 才是模型区分各张图 patch 范围的
依据——这正是 `cat` 天然对应的拼接方式。`batch_size=1` 时单个 2D 张量 `cat` 后形状不变，
和原来 `stack` 出的 `[1, N, D]` 在语义上等价（train_debug.sh 已验证过 `stack` 版本能跑通，
`cat` 版本形状更贴近模型实际期望的输入格式），同时天然支持 `batch_size > 1` 的混合分辨率
场景，不需要再维护一个容易出错的条件分支。

---

### H3：重新生成 `attention_mask`，并传给训练/评估阶段的前向计算

**问题**：`template.py` 的 `add_answer_to_batch`/`add_multiturn_answer` 拼接答案后主动删除了
`attention_mask`（因为长度和拼接后的 `input_ids` 对不上），但 `collate_fn` 从未重新生成它，
`trainer.py`/`evaluator.py` 的前向/生成调用里也从未传过这个参数——最终模型训练和评估时
完全没有 `attention_mask`，batch 内 padding 部分可能被错误地当作有效内容参与 attention 计算。

**修改**：
1. `data/base_dataset.py` 的 `collate_fn`：基于 padding/截断后的 `input_ids` 重新生成
   `attention_mask = (input_ids != pad_token_id).long()`，加入返回的 `result` 字典。
2. `trainer/trainer.py`：训练循环的 `forward_dict` 加入 `attention_mask`。
3. `trainer/evaluator.py`：`evaluate_screenspot` 的生成参数、`evaluate_training_data` 的
   前向调用都加入 `attention_mask`。

**为什么这么改**：`attention_mask` 从"被删除"到"重新生成"再到"传给模型"，这三步缺一不可——
只在 `collate_fn` 里生成而不往下传，等于没修；这是文档里指出的、`batch_size > 1`（有真实
padding）时才会显现的静默正确性问题，`batch_size=1`（当前调试脚本用的配置）不会暴露这个
问题，所以此前一直没被发现。

---

## 第二批：Medium + Low

### M1：`sdpa` 模式不再被意外关闭加速后端

**修改**：`trainer/train.py` 里关闭 `enable_mem_efficient_sdp`/`enable_flash_sdp` 的条件从
`in ["eager", "sdpa"]` 收窄为 `== "eager"`。

**为什么**：选 `sdpa` 就是为了让它自动挑选最优的底层实现，连着它的加速开关一起关掉是
自相矛盾的写法，只有明确选择最朴素的 `eager` 实现时才应该关闭加速后端。

### M2：评估阶段不再用裸 `except` 吞掉所有异常

**修改**：`trainer/evaluator.py` 的 `evaluate_screenspot` 中，将裸 `except:` 改为只捕获
`(ValueError, SyntaxError, TypeError)`（对应"模型输出格式不规整、解析失败"这类预期内的情况），
并用 `logging.warning` 记录具体样本 id 和原始预测文本；预期外的异常（比如数据字段缺失导致的
`KeyError`）不再被静默吞掉。

**为什么**："模型输出解析失败"和"代码/数据本身有 bug"是性质完全不同的两类问题，前者
应该被容忍并计入失败样本，后者应该被开发者感知到，裸 `except` 把两者混为一谈，容易让
代码层面的 bug 被长期误判为"模型能力不足"。

### M3：推理脚本 `smart_resize` 补齐输入校验

**修改**：`inference.py` 的 `_smart_resize` 改为直接调用共享实现（见 M5），自动获得
"图片过小""长宽比过于极端"这两处校验，和训练/评估脚本行为一致。

### M4：推理脚本坐标解析鲁棒性对齐评估脚本

**修改**：`inference.py` 的 `predict()` 在 `ast.literal_eval` 之前，先用共享的
`extract_coordinates` 正则从生成文本中提取 `[x, y]` 子串，能够容忍模型输出中夹杂自然语言
修饰（比如"点击位置是[512, 300]"），和 `evaluator.py` 的容错能力对齐。

**为什么 M3、M4 一起改**：这两个问题的根源相同——推理脚本为了保持"轻量部署、不依赖训练
框架"独立实现了一份逻辑，但独立维护导致它比内部评估脚本更脆弱，属于反常现象；
修复方式就是 M5 的共享模块方案。

### M5：抽取共享工具模块 `utils/qwen_vl_common.py`

**修改**：新增 `utils/qwen_vl_common.py`，集中存放：
- `smart_resize`（图像智能 resize，含合法性校验）
- `point_to_qwen_coords` / `qwen_coords_to_point`（原来分散实现的三份坐标转换逻辑：
  `train_dataset.py._convert_point_for_qwen`、`evaluator.py.convert_point_from_qwen_format`、
  `inference.py._convert_point_to_original`）
- `CHAT_TEMPLATE` 常量（原来在 `train.py` 和 `inference.py` 各硬编码一份）
- `extract_coordinates` 正则提取函数

`data/train_dataset.py`、`trainer/evaluator.py`、`inference/inference.py`、`trainer/train.py`
均改为从这个模块导入，不再各自维护一份。该模块只依赖标准库 `math`/`re`，不引入
`torch`/`deepspeed`/`wandb` 等重型依赖，`inference.py` 仍然保持"轻量部署"的特性。

**验证**：重构后写了一个本地回归测试脚本，用 2000 组随机生成的图片尺寸/坐标，
逐一对比重构前后的计算结果，**逐位完全一致**（包括边界校验触发 `ValueError` 的情况），
确认这是一次纯粹的"消除重复代码"，没有改变任何实际计算逻辑。

### L1：删除未使用的 `BitsAndBytesConfig` import

**修改**：`trainer/train.py` 删除这个从未被实例化、也从未传给 `from_pretrained` 的 import。

**为什么**：当前没有 QLoRA/量化训练的计划，保留一个未使用的 import 只会让后来的维护者
误以为项目已经支持量化训练，删掉更清晰；未来真要支持 QLoRA 时再补上完整的参数解析和
加载逻辑。

### L2：删除 `model_utils.py` 中未被调用的工具函数

**修改**：删除 `count_parameters`、`print_trainable_parameters`、`freeze_module`、
`unfreeze_module` 四个函数（连带更新 `models/__init__.py` 的导出列表）。经全仓库搜索确认，
`train.py` 打印可训练参数用的是 PEFT 自带的 `model.print_trainable_parameters()` 方法，
冻结视觉编码器用的是手写循环，这四个自定义函数从未被实际调用过。

**为什么选择删除而不是回填调用**：这四个函数本身逻辑没问题，但 `train.py` 里对应的
手写逻辑已经跑通并经过 debug 训练验证，把它们替换成调用这几个工具函数不会带来任何
功能收益，只会引入新的、未经验证的改动点；删除未使用的死代码是风险更低的选择。

### L3：为「有意为之的随机性」补充说明注释

**修改**：`data/train_dataset.py._get_sample` 中，在任务类型选择、多轮元素选择这两处
未绑定 `idx` 随机种子的地方补充注释，说明这是有意为之（只有单轮模式的元素选择需要
强可复现性以便调试，任务类型选择和多轮元素选择允许有随机性以增加数据多样性），
不是遗漏，避免后续维护者误判为 bug 而"修复"掉这个有意为之的设计。

**为什么不改成完全确定性**：没有看到"训练数据完全可复现"这一诉求的实际证据，
贸然让所有随机决策都绑定 `idx` 种子，会改变现有的数据采样分布/多样性，属于没有明确
需求驱动的行为变更，风险大于收益，所以选择"记录设计意图"而不是"改变行为"。

---

## 改完之后，你需要在服务器上验证的事情

本地只做了：`python3 -m py_compile` 语法检查、`utils/qwen_vl_common.py` 的纯函数与原实现
做了 2000 组随机输入的回归比对（结果完全一致）。以下这些必须在有 GPU、有 checkpoint/数据集
的服务器上才能验证：

1. `bash scripts/train_debug.sh` 能正常跑完，loss 正常下降。
2. 观察 debug 输出中 point 任务的 answer（可以临时加一行 print `answer_xy`），确认
   现在单轮/多轮产出的坐标数值量级一致（都是 resize 后图片尺寸级别的整数，比如几百到
   一千多），而不是一个 0~1 小数、一个几百的整数混在一起。
3. `model.print_trainable_parameters()` 打印结果里确认没有 `lm_head`。
4. 用 `--train_ratio "3,1,1" --random_sample --record_sample` 跑一下，确认能看到新加的
   `[警告]` 输出。
5. 如果条件允许，起一次小规模的 `--batch_size 2` 训练（哪怕只是 debug 规模），验证
   H2（`pixel_values` cat 拼接）在真实多分辨率 batch 下不报错。
