# ChatGLM3-6B 医疗文本微调方案

本项目提供了一套使用清华大学与智谱AI联合开发的ChatGLM3-6B模型进行医疗领域微调的完整解决方案。作为完全开源的大型双语对话模型，ChatGLM3-6B在中文处理能力和模型开放性上都有很大优势。

## 为什么选择ChatGLM3-6B

- **完全开源**：模型权重、训练代码和推理代码完全开源
- **中文优化**：专为中文场景优化，对医疗专业术语理解更好
- **模型规模适中**：6B参数量在保证性能的同时，能在消费级显卡上运行
- **强大的对话能力**：采用对话式设计，适合问答场景使用
- **丰富的开源工具链**：拥有活跃社区和丰富的微调工具支持

## 文件说明

- `chatglm_finetuning.py`: ChatGLM3-6B模型微调主程序
- `chatglm_finetuning_tst.py`: 微调前后模型对比测试程序
- `financial_data.txt`: 经过优化的医疗领域文本数据集
- `README_chatglm_finetuning.md`: 本文档

## 环境需求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- peft 0.4.0+
- bitsandbytes 0.39.0+
- accelerate 0.20.0+
- CUDA支持的GPU (推荐8GB+显存)

## 安装依赖

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install bitsandbytes>=0.39.0
pip install accelerate>=0.20.0
pip install datasets
```

## 使用方法

### 1. 模型微调

运行以下命令开始微调ChatGLM3-6B模型:

```bash
python chatglm_finetuning.py
```

微调后的模型将保存在指定的保存路径中（默认为"/home/hiar/tazzhang/modelscope/hub/ZhipuAI/chatglm3-6b-finetuned"）。

### 2. 模型测试与对比

运行以下命令测试微调前后的模型效果:

```bash
python chatglm_finetuning_tst.py
```

测试结果将显示在控制台，并记录到`chatglm_comparison.log`文件中。

## 技术细节

### LoRA微调方案

我们采用参数高效微调技术(PEFT)中的LoRA方法对ChatGLM3-6B进行微调，主要优势：

1. **显存高效**：仅需训练少量参数，可在单张消费级GPU上运行
2. **训练速度快**：相比全参数微调，训练速度提升数十倍
3. **保留原始能力**：避免灾难性遗忘，保留模型原有知识
4. **适配性好**：适用于各种硬件配置，可根据GPU显存调整参数

### 4位量化

采用4位量化技术降低模型内存占用：

- 将FP16/FP32参数量化为INT4格式，显存占用降低75%以上
- 仅在推理阶段使用量化，不影响训练精度
- 结合LoRA技术，只对非LoRA参数进行量化

### 数据格式与处理

针对医疗领域问答数据的特殊处理：

1. **数据格式**：
   - 输入数据格式为：`简短问题? 详细问题描述,回答`
   - 例如：`高血压患者能吃党参吗？ 我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？,高血压病人可以口服党参的。党参有降血脂...`

2. **数据处理流程**：
   - 分割简短问题（问号前部分）和剩余内容
   - 从剩余内容中分割出详细问题描述和回答（使用逗号分隔）
   - 合并简短问题和详细描述作为完整问题
   - 将问答对转换为ChatGLM3对话格式：`[Round 1]\n\n问：{问题}\n\n答：{回答}`

3. **边缘情况处理**：
   - 适当处理没有问号的数据
   - 处理可能没有逗号分隔符的情况
   - 保证所有数据都能正确格式化为ChatGLM3的训练格式

4. **分词与截断**：
   - 使用ChatGLM3原生分词器处理文本
   - 设定最大序列长度为1024，超长文本自动截断
   - 自动添加适当填充，确保批处理一致性

## 微调参数说明

- **LoRA配置**：
  - r=16: 低秩矩阵的秩
  - alpha=32: 缩放因子
  - dropout=0.1: 防止过拟合
  - 目标模块: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

- **训练参数**：
  - 学习率: 1e-4
  - 训练轮次: 3
  - 批次大小: 根据GPU显存自适应(通常为1)
  - 梯度累积: 16步（保持有效批量大小）
  - 预热比例: 3%
  - 使用混合精度(fp16)训练提高效率
  - 启用梯度检查点以节省内存

## 模型测试方法分析

### 测试流程

`chatglm_finetuning_tst.py`实现了一套完整的模型对比测试方法：

1. **模型加载**：分别加载基础模型和微调后的模型
2. **生成对比**：使用相同的提示词生成回答
3. **指标计算**：对生成结果计算多种对比指标
4. **结果展示**：输出对比结果和详细评价指标

### 评估指标

- **输出长度比较**：对比微调前后模型输出的文本长度及变化率
- **医疗术语密度**：计算输出中医疗术语的使用频率
- **与提示词相关性**：测量输出与输入提示词的词汇重叠度
- **生成耗时**：记录模型生成文本所需时间

### 测试方法有效性分析

**优点**：
- 直观对比微调前后的差异
- 多维度评估模型变化
- 详细的日志记录便于分析
- 性能指标完整，包括速度和内存使用
- 代码结构清晰，便于扩展
- 适应有限计算资源的环境

**局限性**：
- 测试样本量较小（目前仅一个示例）
- 缺少基于标准NLP指标的客观评估（如BLEU、ROUGE等）
- 医疗术语密度计算方式简单，可能不够全面
- 缺少人工评估对生成质量的主观判断
- 对真实场景下的整体表现预测有限

### 改进建议

为提高测试有效性，可考虑：
- 扩充测试集，包含更多样化的医疗问题
- 引入专业医疗知识评估标准
- 添加更复杂的语义相似度计算
- 设计A/B测试收集用户偏好
- 针对特定医疗子领域（如慢性病、急诊等）进行分类评估

## 代码实现特点

- **GPU设备管理**：通过环境变量和CUDA设置精确控制使用的GPU设备
- **内存优化**：采用多种技术降低内存占用，包括4位量化、梯度检查点和梯度累积
- **错误处理**：完善的异常捕获和日志记录，便于调试和问题追踪
- **模块化设计**：清晰的函数划分，方便扩展和修改
- **自适应计算**：根据可用硬件资源自动调整批处理大小和训练参数

## 注意事项

- 首次下载ChatGLM3-6B模型可能需要较长时间
- 训练和推理过程需要GPU支持，最低建议8GB显存
- 使用4位量化可在4GB显存的GPU上进行推理，但可能影响生成质量
- 为避免微调过度，建议控制训练轮次，防止过拟合
- 确保输入数据格式符合要求，避免数据处理错误
- 测试结果解读需结合实际应用场景，单一指标可能不足以全面评估模型性能

## 参考资料

- [ChatGLM3-6B官方仓库](https://github.com/THUDM/ChatGLM3)
- [PEFT文档](https://huggingface.co/docs/peft/index)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [Transformers文档](https://huggingface.co/docs/transformers/index) 