import os
# 设置环境变量，限制只使用CUDA设备3（设置在最开始，确保其他import前生效）
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging
import torch
import time

# 确保使用单个GPU
torch.cuda.set_device(0)  # 重置设备索引为0（因为只有一个可见设备）

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatglm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 检查必要的依赖是否安装
required_packages = ["transformers", "datasets", "peft", "bitsandbytes", "accelerate"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("错误: 缺少以下必要的依赖库:")
    for package in missing_packages:
        print(f"  - {package}")
    print("\n请使用以下命令安装缺失的依赖:")
    print(f"pip install {' '.join(missing_packages)}")
    exit(1)

# 导入所需的库
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import bitsandbytes as bnb
from accelerate import Accelerator

def check_environment():
    """检查环境设置与依赖"""
    try:
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
            logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
            logger.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        # 检查是否已安装所需的库
        required_packages = ["transformers", "datasets", "peft", "bitsandbytes", "accelerate"]
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"{package} 已安装")
            except ImportError:
                logger.warning(f"{package} 未安装，请先安装此依赖")
        
        return True
    except Exception as e:
        logger.error(f"环境检查时出错: {str(e)}")
        return False

def load_model_and_tokenizer(model_name="THUDM/chatglm3-6b"):
    """
    加载ChatGLM3-6B模型和Tokenizer
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    try:
        logger.info(f"正在加载模型和分词器: {model_name}...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # Monkey-patch ChatGLMTokenizer._pad to ignore unsupported padding_side argument
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        tokenizer._pad = lambda encoded_inputs, *args, **kwargs: PreTrainedTokenizerBase._pad(
            tokenizer,
            encoded_inputs,
            *args,
            **{k: v for k, v in kwargs.items() if k != "padding_side"}
        )
        
        # 指定单个GPU设备，避免模型被分布到多个GPU上
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 现在使用cuda:0，因为在环境变量中设置了CUDA_VISIBLE_DEVICES=3
        logger.info(f"将模型加载到单个设备上: {device}")
        
        # 为了节省显存，使用4位量化和LoRA微调，但指定单个设备
        # 设置计算类型为float16，与模型类型一致，避免警告
        compute_dtype = torch.float16
        
        # 创建BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": device},  # 使用单个设备
            torch_dtype=compute_dtype,
            quantization_config=bnb_config
        )
        
        # 准备模型进行LoRA微调
        model = prepare_model_for_kbit_training(model)
        
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,         # LoRA注意力维度
            lora_alpha=32,  # LoRA缩放因子
            lora_dropout=0.1,  # LoRA丢弃率
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 要微调的模块
        )
        
        # 获取PEFT模型
        model = get_peft_model(model, peft_config)
        
        # 打印模型中可训练的参数
        model.print_trainable_parameters()
        
        logger.info("模型和分词器加载成功")
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载模型和分词器时出错: {str(e)}")
        raise

def prepare_dataset(file_path, tokenizer, max_length=1024):
    """
    加载并预处理数据
    
    Args:
        file_path: 数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        dataset: 处理后的数据集
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件 {file_path} 不存在")
            
        logger.info(f"正在加载数据集: {file_path}")
        
        # 使用datasets库加载文本文件
        dataset = load_dataset('text', data_files={'train': file_path})
        logger.info(f"数据集加载成功，共 {len(dataset['train'])} 条样本")
        
        # ChatGLM3格式化数据
        def format_example(example):
            # 将每行内容转换为问答对
            text = example["text"].strip()
            
            # 按照问题和答案通过空格分割的格式处理
            seq = ","
            if seq in text:
                # 使用第一个空格分割问题和答案
                question, answer = text.split(seq, 1)
                
                # 构建ChatGLM3对话格式
                formatted_text = f"[Round 1]\n\n问：{question}\n\n答：{answer}"
            else:
                # 处理没有空格的情况
                formatted_text = f"[Round 1]\n\n问：请介绍一下\"{text[:16]}...\"的相关内容\n\n答：{text}"
            
            logger.info(f"格式化后的文本: {formatted_text}")
            return {"formatted_text": formatted_text}
        
        # 格式化数据集
        formatted_dataset = dataset.map(format_example)
        
        # 分词函数
        def tokenize_function(examples):
            """将文本转换为token"""
            tokenized_inputs = tokenizer(
                examples["formatted_text"],
                padding=True,  # 修改为简单的padding=True
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 创建标签（与输入相同，用于自回归训练）
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            return tokenized_inputs
        
        logger.info("开始对数据集进行分词处理...")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text", "formatted_text"],
            desc="对数据集进行分词处理"
        )
        logger.info("数据集分词处理完成")
        
        # 划分训练集和验证集
        split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        
        return split_dataset
    except Exception as e:
        logger.error(f"准备数据集时出错: {str(e)}")
        raise

def train_model(model, tokenizer, dataset, output_dir="./chatglm-finetuned"):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        tokenizer: 分词器
        dataset: 数据集
        output_dir: 输出目录
        
    Returns:
        model: 训练后的模型
    """
    try:
        logger.info("配置训练参数...")
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 获取模型所在设备
        device = next(model.parameters()).device
        logger.info(f"模型位于设备: {device}")
        
        # 计算设备自适应
        if torch.cuda.is_available():
            batch_size = 1  # 减小batch_size以避免内存不足
            gradient_accumulation_steps = 16  # 增加梯度累积步数以保持有效批量大小
        else:
            batch_size = 1
            gradient_accumulation_steps = 16
        
        # 训练参数配置 - 移除不支持的evaluation_strategy
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=1e-4,   # 使用较小的学习率进行微调
            weight_decay=0.01,    # 正则化参数
            warmup_ratio=0.03,    # 预热比例
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            # 移除evaluation_strategy参数
            save_total_limit=3,   # 最多保存3个检查点
            # 移除load_best_model_at_end，因为没有评估策略时不能使用
            report_to="none",     # 不使用外部报告系统
            fp16=True,            # 使用混合精度训练
            remove_unused_columns=False,  # ChatGLM3需要保留某些列
            # 指定设备
            no_cuda=False,
            dataloader_drop_last=True,
            dataloader_num_workers=0,  # 减少数据加载器工作线程数量
            gradient_checkpointing=True,  # 启用梯度检查点以节省内存
            # 禁用分布式训练
            local_rank=-1,
            ddp_backend=None
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            # 移除eval_dataset，因为没有设置评估策略
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # 开始训练
        logger.info("开始训练模型...")
        trainer.train()
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"模型训练完成，总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        
        return model
    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        raise

def save_model(model, tokenizer, save_path):
    """
    保存模型和分词器
    
    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        save_path: 保存路径
    """
    try:
        logger.info(f"正在将模型保存到 {save_path}...")
        
        # 创建保存目录（如果不存在）
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重（只保存LoRA权重）
        model.save_pretrained(save_path)
        
        # 保存分词器
        tokenizer.save_pretrained(save_path)
        
        logger.info("模型和分词器保存成功")
    except Exception as e:
        logger.error(f"保存模型时出错: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 检查环境
        if not check_environment():
            logger.error("环境检查失败，请解决问题后重试")
            return
        
        # 数据文件路径
        data_path = "./medicine_QA.txt"

        model_path = "/home/hiar/tazzhang/modelscope/hub/ZhipuAI/chatglm3-6b"
        
        # 模型保存路径
        save_path = "/home/hiar/tazzhang/modelscope/hub/ZhipuAI/chatglm3-6b-finetuned"
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 准备数据集
        dataset = prepare_dataset(data_path, tokenizer)
        
        # 训练模型
        model = train_model(model, tokenizer, dataset, output_dir="./results")
        
        # 保存模型
        save_model(model, tokenizer, save_path)
        
        logger.info("所有任务完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 