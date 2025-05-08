import warnings
warnings.filterwarnings(action='ignore')
import os
import torch
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoProcessor, Blip2ForConditionalGeneration, TrainingArguments, Trainer, Blip2Config,
    BlipImageProcessor, Blip2Processor, GPT2Tokenizer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset

# 设置
torch.cuda.empty_cache()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和处理器
print("正在加载模型和处理器...")
model_folder = os.path.join("/root/autodl-tmp/models", "blip2-vizwizqa")
config = Blip2Config.from_pretrained(model_folder)

# 修改配置中的视觉编码器隐藏维度
config.vision_config.hidden_size = 1408
config.vision_config.intermediate_size = 6144
model = Blip2ForConditionalGeneration.from_pretrained(
    model_folder,
    config=config
).to(device)
image_processor = BlipImageProcessor.from_pretrained(model_folder)

# 加载分词器
print(f"Loading GPT2Tokenizer from: /root/autodl-tmp/models/blip2-vizwizqa")
tokenizer = GPT2Tokenizer.from_pretrained("/root/autodl-tmp/models/blip2-vizwizqa")
# 然后组合它们
processor = Blip2Processor(image_processor, tokenizer)
# model = Blip2ForConditionalGeneration.from_pretrained("/root/autodl-tmp/models/blip2-vizwizqa", device_map="auto")     

# LoRA配置
print("正在应用 LoRA 配置...")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
config = LoraConfig(r=16, 
                    lora_alpha=32, 
                    lora_dropout=0.05, 
                    bias="none", 
                    target_modules=["q_proj", "k_proj"])
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)

# 加载 VizWiz 数据集
print("正在加载 VizWiz 数据集...")
VIZ_TRAIN_IMG_PATH = '/root/autodl-tmp/data/VizWiz/train'
VIZ_VAL_IMG_PATH = '/root/autodl-tmp/data/VizWiz/val'

class VizWizDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, processor, img_path, max_length=32):
        if isinstance(annotation_file, str):
            self.data = Dataset.from_json(annotation_file).to_dict('records')
        else:
            self.data = annotation_file
        self.processor = processor
        self.img_path = img_path
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        image_id = str(data["image_id"]).zfill(12)
        image = Image.open(os.path.join(self.img_path, image_id + '.jpg')).convert("RGB") if os.path.exists(os.path.join(self.img_path, image_id + '.jpg')) else Image.new('RGB', (224, 224), color='black')
        
        question = data['question']
        encoding = self.processor(images=image, text=question, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        answers = data.get('answers', [])
        if answers:
            answer_count = {ans: answers.count(ans) for ans in set(answers)}
            selected_answer = random.choices(answers, weights=[answer_count[ans] for ans in answers])[0]
            answer_tokens = self.processor.tokenizer.encode(selected_answer, add_special_tokens=False, max_length=self.max_length, truncation=True, padding="max_length")
            encoding["labels"] = torch.tensor(answer_tokens)
        else:
            encoding["labels"] = torch.zeros(self.max_length, dtype=torch.long)  # Default empty label if no answer
        
        return encoding

# 加载数据集
vizwiz_train_data = Dataset.from_json('/root/autodl-tmp/data/VizWiz/VizWiz_train.json')
vizwiz_valid_data = Dataset.from_json('/root/autodl-tmp/data/VizWiz/VizWiz_val.json')

vizwiz_train_dataset = VizWizDataset(vizwiz_train_data, processor, VIZ_TRAIN_IMG_PATH)
vizwiz_valid_dataset = VizWizDataset(vizwiz_valid_data, processor, VIZ_VAL_IMG_PATH)

# 加载 VQA 数据集
print("正在加载 VQA 数据集...")
VQA_TRAIN_IMG_PATH = '/root/autodl-tmp/data/VQAv2/train'
VQA_VAL_IMG_PATH = '/root/autodl-tmp/data/VQAv2/val'

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, processor, img_path, max_length=32):
        if isinstance(annotation_file, str):
            self.data = Dataset.from_json(annotation_file).to_dict('records')
        else:
            self.data = annotation_file
        self.processor = processor
        self.img_path = img_path
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        image_id = data["image_id"]
        img_path = os.path.join(self.img_path, f"COCO_train2014_{str(image_id).zfill(12)}.jpg")
        image = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else Image.new('RGB', (224, 224), color='black')
        
        question = data['question']
        encoding = self.processor(images=image, text=question, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        answers = data['answers'] if 'answers' in data and data['answers'] else []
        if answers:
            answer_count = {ans: answers.count(ans) for ans in set(answers)}
            selected_answer = random.choices(answers, weights=[answer_count[ans] for ans in answers])[0]
            answer_tokens = self.processor.tokenizer.encode(selected_answer, add_special_tokens=False, max_length=self.max_length, truncation=True, padding="max_length")
            encoding["labels"] = torch.tensor(answer_tokens)
        else:
            encoding["labels"] = torch.zeros(self.max_length, dtype=torch.long)
        
        return encoding

# 加载 VQA 数据
vqa_train_data = Dataset.from_json('/root/autodl-tmp/data/VQAv2/VQA_train.json')
vqa_valid_data = Dataset.from_json('/root/autodl-tmp/data/VQAv2/VQA_val.json')

vqa_train_dataset = VQADataset(vqa_train_data, processor, VQA_TRAIN_IMG_PATH)
vqa_valid_dataset = VQADataset(vqa_valid_data, processor, VQA_VAL_IMG_PATH)

# 合并 VizWiz 和 VQA 数据集
print("正在合并 VizWiz 和 VQA 数据集...")
combined_train_dataset = ConcatDataset([vizwiz_train_dataset, vqa_train_dataset])
combined_valid_dataset = ConcatDataset([vizwiz_valid_dataset, vqa_valid_dataset])



# 定义评估指标函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 获取前三个预测答案
    top_3_preds = pred.predictions.argsort(-1)[:, :3]  # 按照概率排序，并选取前3个预测
    
    # 对于每个样本，检查是否有至少一个预测与真实答案匹配
    correct_count = 0
    for i in range(len(labels)):
        if labels[i] in top_3_preds[i]:
            correct_count += 1
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    
    # 返回相关的评估指标
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auc,
        'correct_count': correct_count  # 计算前3个预测中有匹配的数量
    }

# 训练参数
combined_training_args = TrainingArguments(
    output_dir='../results', 
    num_train_epochs=3, 
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1, 
    warmup_steps=500, 
    logging_dir='../logs', 
    logging_steps=2000, 
    do_train=True, 
    fp16=True, 
    fp16_opt_level="02", 
    run_name="blip-2_QA_finetuning-Combined", 
    seed=3
)

# Trainer设置
combined_trainer = Trainer(
    model=model, 
    args=combined_training_args, 
    train_dataset=combined_train_dataset, 
    eval_dataset=combined_valid_dataset, 
    compute_metrics=compute_metrics
)

# 开始训练
combined_trainer.train()

# 最终模型保存
print("正在保存最终模型...")
model.save_pretrained("VQA-for-blind-combined", "/root/autodl-tmp/BLIP/output")

print("训练完成！")

results = combined_trainer.evaluate(eval_dataset=combined_valid_dataset)

# 输出评估结果
print("Evaluation results:")
for key, value in results.items():
    print(f"{key}: {value}")
