import os
# 设置环境变量，限制只使用CUDA设备3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging
import torch
import time
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
from rouge import Rouge

# 确保使用单个GPU
torch.cuda.set_device(0)  # 重置设备索引为0（因为只有一个可见设备）

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatglm_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_device():
    """获取可用的设备（CPU/GPU）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    if device == "cuda":
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        logger.info(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    else:
        logger.info("未检测到GPU，使用CPU进行推理")
    return device

def load_base_model(model_name="THUDM/chatglm3-6b", device=None):
    """
    加载基础版ChatGLM3-6B模型
    
    Args:
        model_name: 模型名称或路径
        device: 运行设备 (cpu/cuda)
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    try:
        logger.info(f"正在加载基础版模型: {model_name}...")
        
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
        
        # 加载模型
        model_loading_args = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
        }
        
        # 根据设备决定加载方式
        if device == "cuda":
            model_loading_args["torch_dtype"] = torch.float16
            model_loading_args["device_map"] = {"": 0}  # 直接指定设备映射到cuda:0
        
        # 记录模型加载开始时间
        start_time = time.time()
        model = AutoModel.from_pretrained(**model_loading_args)
        load_time = time.time() - start_time
        
        # 如果不使用device_map，则需要手动移动模型到设备
        if device == "cuda" and "device_map" not in model_loading_args:
            model = model.to(device)
            
        logger.info(f"基础版模型加载成功，加载耗时: {load_time:.2f}秒")
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载基础版模型时出错: {str(e)}")
        raise

def load_finetuned_model(base_model_name="THUDM/chatglm3-6b", adapter_path="./models/chatglm3-6b-finetuned", device=None):
    """
    加载微调后的ChatGLM3-6B模型
    
    Args:
        base_model_name: 基础模型名称或路径
        adapter_path: LoRA适配器路径
        device: 运行设备 (cpu/cuda)
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    try:
        logger.info(f"正在加载微调模型，基础模型: {base_model_name}，适配器: {adapter_path}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
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
        
        # 加载基础模型
        model_loading_args = {
            "pretrained_model_name_or_path": base_model_name,
            "trust_remote_code": True,
        }
        
        # 根据设备决定加载方式
        if device == "cuda":
            model_loading_args["torch_dtype"] = torch.float16
            model_loading_args["device_map"] = {"": 0}  # 直接指定设备映射到cuda:0
        
        # 记录模型加载开始时间
        start_time = time.time()
        base_model = AutoModel.from_pretrained(**model_loading_args)
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path
        )
        load_time = time.time() - start_time
        
        # 如果不使用device_map，则需要手动移动模型到设备
        if device == "cuda" and "device_map" not in model_loading_args:
            model = model.to(device)
        
        logger.info(f"微调模型加载成功，加载耗时: {load_time:.2f}秒")
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载微调模型时出错: {str(e)}")
        raise

def generate_response(prompt, model, tokenizer, device=None, max_length=512, temperature=0.7, top_p=0.9):
    """
    使用模型生成回答
    
    Args:
        prompt: 输入提示词
        model: 语言模型
        tokenizer: 分词器
        device: 运行设备 (cpu/cuda)
        max_length: 生成文本的最大长度
        temperature: 温度参数，控制生成文本的随机性
        top_p: 核采样参数
    
    Returns:
        response: 生成的回答
        generation_time: 生成耗时（秒）
    """
    try:
        logger.info(f"使用提示词生成回答: '{prompt}'")
        
        # 记录生成开始时间
        start_time = time.time()
        
        # 记录内存使用情况
        if device == "cuda":
            mem_before = torch.cuda.memory_allocated(0)/1024**3
            logger.info(f"生成前GPU内存使用: {mem_before:.2f} GB")
        
        # 准备输入格式
        prompt_text = f"[Round 1]\n\n问：{prompt}\n\n答："
        
        # 编码输入
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 进行贪婪解码生成
        response_ids = inputs["input_ids"].clone()
        max_new_tokens = min(max_length - inputs["input_ids"].shape[1], 2048)
        
        # 设置起始位置
        position = inputs["input_ids"].shape[1]
        
        # 手动逐个生成token
        for _ in range(max_new_tokens):
            # 截取inputs到当前位置
            current_inputs = {
                "input_ids": response_ids[:, :position],
                "attention_mask": torch.ones_like(response_ids[:, :position])
            }
            
            # 前向传播
            with torch.no_grad():
                outputs = model(**current_inputs)
                
            # 获取下一个token的概率分布
            next_token_logits = outputs.logits[:, -1, :]
            
            # 温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # 只保留概率最高的top_p比例的token
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累计超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将被移除的索引设置为负无穷
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查是否生成了结束符
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 添加新token到序列
            response_ids = torch.cat([response_ids, next_token], dim=-1)
            position += 1
        
        # 解码生成的文本
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "答：" in response_text:
            response = response_text.split("答：", 1)[1].strip()
        else:
            response = response_text
        
        # 记录生成结束时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 记录内存使用情况
        if device == "cuda":
            mem_after = torch.cuda.memory_allocated(0)/1024**3
            logger.info(f"生成后GPU内存使用: {mem_after:.2f} GB，峰值: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
        
        logger.info(f"回答生成耗时: {generation_time:.2f}秒")
        
        return response, generation_time
    except Exception as e:
        logger.error(f"生成回答时出错: {str(e)}")
        raise

def calculate_metrics(base_output, finetuned_output, prompt, expected_answer=None):
    """
    计算生成文本的评价指标
    
    Args:
        base_output: 基础模型输出
        finetuned_output: 微调模型输出
        prompt: 原始提示词
        expected_answer: 预期/标准答案(可选)
        
    Returns:
        metrics: 包含多个评价指标的字典
    """
    try:
        # 计算输出长度差异
        base_len = len(base_output)
        finetuned_len = len(finetuned_output)
        
        # 计算输出与提示词的重叠比例（简单估计相关性）
        prompt_tokens = set(prompt.split())
        base_overlap = len([word for word in base_output.split() if word in prompt_tokens]) / len(prompt_tokens) if prompt_tokens else 0
        finetuned_overlap = len([word for word in finetuned_output.split() if word in prompt_tokens]) / len(prompt_tokens) if prompt_tokens else 0
        
        # 计算医疗术语使用频率
        medical_terms = [
            "高血压", "血压", "症状", "治疗", "病人", "患者", "药物", "医生", "医院", "疾病", 
            "血脂", "心脏", "心血管", "血液", "健康", "检查", "降压", "用药", "药品", "服药",
            "诊断", "复查", "建议", "禁忌", "注意事项", "副作用", "不良反应", "并发症", "慢性病",
            "党参", "中药", "西药", "口服", "忌口", "饮食", "营养", "锻炼", "运动"
        ]
        
        base_medical_term_count = sum(base_output.count(term) for term in medical_terms)
        finetuned_medical_term_count = sum(finetuned_output.count(term) for term in medical_terms)
        
        # 避免除以零的情况
        base_term_density = base_medical_term_count / base_len if base_len > 0 else 0
        finetuned_term_density = finetuned_medical_term_count / finetuned_len if finetuned_len > 0 else 0
        term_density_ratio = finetuned_term_density / base_term_density if base_term_density > 0 else float('inf')
        
        # 计算BLEU分数(如果有标准答案)
        bleu_score_base = 0
        bleu_score_finetuned = 0
        rouge_scores_base = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        rouge_scores_finetuned = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        
        if expected_answer:
            # 准备BLEU评分
            try:
                reference = [word_tokenize(expected_answer)]
                candidate_base = word_tokenize(base_output)
                candidate_finetuned = word_tokenize(finetuned_output)
                
                bleu_score_base = sentence_bleu(reference, candidate_base)
                bleu_score_finetuned = sentence_bleu(reference, candidate_finetuned)
                
                # 计算ROUGE分数
                rouge = Rouge()
                base_scores = rouge.get_scores(base_output, expected_answer)[0]
                finetuned_scores = rouge.get_scores(finetuned_output, expected_answer)[0]
                
                rouge_scores_base = {
                    "rouge-1": base_scores["rouge-1"]["f"],
                    "rouge-2": base_scores["rouge-2"]["f"],
                    "rouge-l": base_scores["rouge-l"]["f"]
                }
                
                rouge_scores_finetuned = {
                    "rouge-1": finetuned_scores["rouge-1"]["f"],
                    "rouge-2": finetuned_scores["rouge-2"]["f"],
                    "rouge-l": finetuned_scores["rouge-l"]["f"]
                }
                
            except Exception as e:
                logger.warning(f"计算BLEU/ROUGE分数时出错: {str(e)}")
        
        # 信息流畅性评估（简单估计：使用标点符号比例）
        base_punctuation_count = len(re.findall(r'[，。！？；：、]', base_output))
        finetuned_punctuation_count = len(re.findall(r'[，。！？；：、]', finetuned_output))
        
        base_fluency = base_punctuation_count / base_len if base_len > 0 else 0
        finetuned_fluency = finetuned_punctuation_count / finetuned_len if finetuned_len > 0 else 0
        
        # 统计专业程度（使用简单估计：医学术语频率与复杂句式使用）
        base_complex_sentence_count = len(re.findall(r'([^，。；！？]+[，；][^，。；！？]+)+[。！？]', base_output))
        finetuned_complex_sentence_count = len(re.findall(r'([^，。；！？]+[，；][^，。；！？]+)+[。！？]', finetuned_output))
        
        base_professionalism = (base_medical_term_count + base_complex_sentence_count) / base_len if base_len > 0 else 0
        finetuned_professionalism = (finetuned_medical_term_count + finetuned_complex_sentence_count) / finetuned_len if finetuned_len > 0 else 0
        
        metrics = {
            # 基本长度指标
            "length_difference": finetuned_len - base_len,
            "base_length": base_len,
            "finetuned_length": finetuned_len,
            "length_ratio": finetuned_len / base_len if base_len > 0 else float('inf'),
            
            # 相关性指标
            "prompt_relevance_base": base_overlap,
            "prompt_relevance_finetuned": finetuned_overlap,
            "prompt_relevance_improvement": finetuned_overlap - base_overlap,
            
            # 医疗术语指标
            "base_medical_term_count": base_medical_term_count,
            "finetuned_medical_term_count": finetuned_medical_term_count,
            "base_medical_term_density": base_term_density,
            "finetuned_medical_term_density": finetuned_term_density,
            "medical_term_density_ratio": term_density_ratio,
            
            # 流畅性和专业性
            "base_fluency": base_fluency,
            "finetuned_fluency": finetuned_fluency,
            "fluency_improvement": finetuned_fluency - base_fluency,
            "base_professionalism": base_professionalism,
            "finetuned_professionalism": finetuned_professionalism,
            "professionalism_improvement": finetuned_professionalism - base_professionalism,
            
            # 标准评估指标(如有参考答案)
            "bleu_score_base": bleu_score_base,
            "bleu_score_finetuned": bleu_score_finetuned,
            "bleu_improvement": bleu_score_finetuned - bleu_score_base,
            "rouge_scores_base": rouge_scores_base,
            "rouge_scores_finetuned": rouge_scores_finetuned
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算评估指标时出错: {str(e)}")
        # 提供一个基本的度量标准，即使出错
        return {
            "length_difference": finetuned_len - base_len,
            "error_in_metrics": str(e)
        }

def compare_models(prompts, base_model_name="THUDM/chatglm3-6b", finetuned_model_path="./models/chatglm3-6b-finetuned", expected_answers=None):
    """
    比较基础模型和微调模型的表现
    
    Args:
        prompts: 提示词列表
        base_model_name: 基础模型名称或路径
        finetuned_model_path: 微调模型适配器路径
        expected_answers: 预期答案字典，键为提示词
        
    Returns:
        results: 包含各模型生成结果的字典
        metrics: 评价指标的字典
        memory_stats: 内存使用统计
    """
    # 获取设备
    device = get_device()
    
    results = {}
    all_metrics = {}
    memory_stats = {
        "initial_memory": 0,
        "peak_memory": 0,
        "after_test_memory": 0
    }
    
    try:
        # 记录初始内存使用
        if device == "cuda":
            memory_stats["initial_memory"] = torch.cuda.memory_allocated(0)/1024**3
            logger.info(f"初始GPU内存使用: {memory_stats['initial_memory']:.2f} GB")
        
        # 加载基础模型
        base_model, base_tokenizer = load_base_model(base_model_name, device)
        
        # 加载微调模型
        finetuned_model, finetuned_tokenizer = load_finetuned_model(base_model_name, finetuned_model_path, device)
        
        # 对每个提示词进行测试
        for i, prompt in enumerate(prompts):
            logger.info("="*50)
            logger.info(f"测试提示词({i+1}/{len(prompts)}): '{prompt}'")
            logger.info("="*50)
            
            # 获取预期答案(如果有)
            expected_answer = None
            if expected_answers and prompt in expected_answers:
                expected_answer = expected_answers[prompt]
            
            # 使用基础模型生成回答
            base_output, base_time = generate_response(prompt, base_model, base_tokenizer, device)
            logger.info(f"基础模型输出: {base_output}")
            
            # 使用微调模型生成回答
            finetuned_output, finetuned_time = generate_response(prompt, finetuned_model, finetuned_tokenizer, device)
            logger.info(f"微调模型输出: {finetuned_output}")
            
            # 计算指标
            metrics = calculate_metrics(base_output, finetuned_output, prompt, expected_answer)
            logger.info(f"评价指标: {metrics}")
            
            # 保存结果
            results[prompt] = {
                "base_model": {
                    "output": base_output,
                    "time": base_time
                },
                "finetuned_model": {
                    "output": finetuned_output,
                    "time": finetuned_time
                }
            }
            
            all_metrics[prompt] = metrics
            
            # 记录峰值内存
            if device == "cuda":
                current_peak = torch.cuda.max_memory_allocated(0)/1024**3
                memory_stats["peak_memory"] = max(memory_stats["peak_memory"], current_peak)
                logger.info(f"当前峰值GPU内存使用: {current_peak:.2f} GB")
            
            logger.info("-"*50)
            
        # 释放GPU内存
        del base_model, base_tokenizer, finetuned_model, finetuned_tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            memory_stats["after_test_memory"] = torch.cuda.memory_allocated(0)/1024**3
            logger.info(f"测试后GPU内存使用: {memory_stats['after_test_memory']:.2f} GB")
            
        return results, all_metrics, memory_stats
    
    except Exception as e:
        logger.error(f"比较模型时出错: {str(e)}")
        raise

def print_comparison_results(results, metrics, memory_stats=None):
    """打印对比结果和评价指标"""
    print("\n" + "="*100)
    print("模型对比测试结果")
    print("="*100)
    
    if memory_stats:
        print(f"\n【内存使用统计】")
        print(f"初始GPU内存: {memory_stats['initial_memory']:.2f} GB")
        print(f"峰值GPU内存: {memory_stats['peak_memory']:.2f} GB")
        print(f"测试后GPU内存: {memory_stats['after_test_memory']:.2f} GB")
        print("-"*100)
    
    # 计算平均指标
    if metrics:
        avg_metrics = {
            "avg_length_ratio": sum(m.get("length_ratio", 0) for m in metrics.values()) / len(metrics),
            "avg_prompt_relevance_improvement": sum(m.get("prompt_relevance_improvement", 0) for m in metrics.values()) / len(metrics),
            "avg_medical_term_density_ratio": sum(m.get("medical_term_density_ratio", 0) for m in metrics.values() if m.get("medical_term_density_ratio", 0) != float('inf')) / len(metrics),
            "avg_professionalism_improvement": sum(m.get("professionalism_improvement", 0) for m in metrics.values()) / len(metrics),
            "avg_base_time": sum(results[p]["base_model"]["time"] for p in results) / len(results),
            "avg_finetuned_time": sum(results[p]["finetuned_model"]["time"] for p in results) / len(results),
        }
        
        print(f"\n【平均指标】")
        print(f"输出长度比例: {avg_metrics['avg_length_ratio']:.2f}")
        print(f"相关性改进: {avg_metrics['avg_prompt_relevance_improvement']:.4f}")
        print(f"医疗术语密度比例: {avg_metrics['avg_medical_term_density_ratio']:.2f}")
        print(f"专业性改进: {avg_metrics['avg_professionalism_improvement']:.4f}")
        print(f"生成时间 - 基础模型: {avg_metrics['avg_base_time']:.2f}秒, 微调模型: {avg_metrics['avg_finetuned_time']:.2f}秒")
        print("-"*100)
    
    for prompt, outputs in results.items():
        print(f"\n【提示词】: {prompt}")
        print("-"*100)
        
        base_output = outputs["base_model"]["output"]
        finetuned_output = outputs["finetuned_model"]["output"]
        
        print(f"【基础模型】(生成耗时: {outputs['base_model']['time']:.2f}秒):")
        print(base_output)
        print("-"*50)
        
        print(f"【微调模型】(生成耗时: {outputs['finetuned_model']['time']:.2f}秒):")
        print(finetuned_output)
        print("-"*50)
        
        # 打印指标
        m = metrics[prompt]
        print("【评价指标】:")
        print(f"- 输出长度: 基础模型 {m['base_length']} vs 微调模型 {m['finetuned_length']} (变化率: {m['length_ratio']:.2f})")
        print(f"- 医疗术语密度: 基础模型 {m['base_medical_term_density']:.4f} vs 微调模型 {m['finetuned_medical_term_density']:.4f} (变化率: {m['medical_term_density_ratio']:.2f})")
        print(f"- 与提示词相关性: 基础模型 {m['prompt_relevance_base']:.4f} vs 微调模型 {m['prompt_relevance_finetuned']:.4f} (改进: {m['prompt_relevance_improvement']:.4f})")
        print(f"- 流畅性: 基础模型 {m['base_fluency']:.4f} vs 微调模型 {m['finetuned_fluency']:.4f} (改进: {m['fluency_improvement']:.4f})")
        print(f"- 专业性: 基础模型 {m['base_professionalism']:.4f} vs 微调模型 {m['finetuned_professionalism']:.4f} (改进: {m['professionalism_improvement']:.4f})")
        
        # 如果有BLEU分数，打印它们
        if 'bleu_score_base' in m and m['bleu_score_base'] > 0:
            print(f"- BLEU分数: 基础模型 {m['bleu_score_base']:.4f} vs 微调模型 {m['bleu_score_finetuned']:.4f} (改进: {m['bleu_improvement']:.4f})")
            
            # 打印ROUGE分数
            if m['rouge_scores_base']['rouge-1'] > 0:
                print(f"- ROUGE-1: 基础模型 {m['rouge_scores_base']['rouge-1']:.4f} vs 微调模型 {m['rouge_scores_finetuned']['rouge-1']:.4f}")
                print(f"- ROUGE-2: 基础模型 {m['rouge_scores_base']['rouge-2']:.4f} vs 微调模型 {m['rouge_scores_finetuned']['rouge-2']:.4f}")
                print(f"- ROUGE-L: 基础模型 {m['rouge_scores_base']['rouge-l']:.4f} vs 微调模型 {m['rouge_scores_finetuned']['rouge-l']:.4f}")
                
        print("-"*100)
    
    print("\n")
    
    # 打印测试方法评估
    print("\n" + "="*100)
    print("测试方法有效性分析")
    print("="*100)
    
    print("\n【优点】:")
    advantages = [
        "1. 直观对比: 同时展示微调前后的输出，便于直观比较效果差异",
        "2. 多维度评估: 从长度、相关性、专业术语、流畅性等多个角度评估模型变化",
        "3. 详细的日志: 记录完整的测试过程，包括时间和内存使用情况，便于分析",
        "4. 资源监控: 跟踪GPU内存使用和生成时间，确保资源效率",
        "5. 模块化设计: 代码结构清晰，各功能独立封装，便于扩展",
        "6. 错误处理: 完善的异常捕获机制，提高测试稳定性"
    ]
    for adv in advantages:
        print(adv)
    
    print("\n【局限性】:")
    limitations = [
        "1. 样本量有限: 测试样本数量较少，可能不足以全面评估模型能力",
        "2. 评估标准简化: 专业性、流畅性等指标使用简单规则计算，缺乏完整的语言学评估",
        "3. 缺少人工评估: 没有医学专家对生成内容质量的主观评价",
        "4. 标准答案缺失: 部分测试没有标准参考答案，影响BLEU/ROUGE等客观评分",
        "5. 特定领域局限: 评估偏向医疗领域，对其他领域泛化能力评估不足"
    ]
    for lim in limitations:
        print(lim)
    
    print("\n【改进建议】:")
    improvements = [
        "1. 扩充测试集: 增加更多样化的医疗问题，涵盖不同病种和复杂程度",
        "2. 引入专业评估: 邀请医学专家进行专业质量评估，建立更可靠的评分标准",
        "3. 增强客观指标: 添加更多NLP标准评估指标，如困惑度(Perplexity)和语义相似度",
        "4. 构建标准答案库: 为测试问题准备高质量标准答案，提高评估准确性",
        "5. 分类评估能力: 按照不同医疗子领域(如慢性病管理、急诊指导等)分类评估模型表现"
    ]
    for imp in improvements:
        print(imp)
    
    print("\n")

def main():
    """主函数"""
    try:
        # 测试提示词
        prompts = [
            "高血压患者能吃党参吗",
        ]
        
        # 预期答案(示例)
        expected_answers = {
            "高血压患者能吃党参吗？": "高血压病人可以口服党参的。党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。"
        }
        
        # 模型路径
        base_model_name = "/home/hiar/tazzhang/modelscope/hub/ZhipuAI/chatglm3-6b"     
        finetuned_model_path = "/home/hiar/tazzhang/modelscope/hub/ZhipuAI/chatglm3-6b-finetuned"
        
        # 比较两个模型
        logger.info(f"开始测试，共{len(prompts)}个提示词")
        results, metrics, memory_stats = compare_models(prompts, base_model_name, finetuned_model_path, expected_answers)
        
        # 打印结果
        print_comparison_results(results, metrics, memory_stats)
        
        logger.info("所有测试完成")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 