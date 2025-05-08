from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
from inference_utils import *

# 路径设置保持不变
sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer')
sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer/fairseq')
from demo import visual_grounding

vizwiz_data_base_path = '/root/dataset-split/split'
viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'vizwiz_val_images')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'vizwiz_val_annotations.json')

pretrained_model = 'vilt-b32-finetuned-vqa'
model_folder = os.path.join('/root/autodl-tmp/models', pretrained_model)
finetune_folder = '/root/autodl-tmp/output/models/custom_vqa_vilt-b32-finetuned-vqa'

processor = ViltProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

missing_words_in_vocab = []

COMBINED_IMAGE_DIR = '/root/dataset-split/split/vizwiz_val_images'
question_image_data = '/root/dataset-split/split/vizwiz_val_annotations.json'
question_image_data = json.load(open(question_image_data))
print(type(question_image_data))
print(question_image_data[0])

# 提取所有问题到一个新的列表
questions = [item['question'] for item in question_image_data]
print(f"总问题数量: {len(questions)}")

# 创建输出目录
os.makedirs('/root/autodl-tmp/output/predictions', exist_ok=True)

# 定义输出文件路径
stage1_output_file = '/root/autodl-tmp/output/predictions/stage1_vilt_results_yuanshi.json'
stage2_output_file = '/root/autodl-tmp/output/predictions/stage2_grounding_results_yuanshi.json'
stage3_output_file = '/root/autodl-tmp/output/predictions/stage3_final_results_yuanshi.json'
all_stages_output_file = '/root/autodl-tmp/output/predictions/all_stages_results_yuanshi.json'

# 单个样本处理（用于测试和展示）
def process_single_samples(start_idx=111, num_samples=20):
    """
    处理指定数量的样本，输出每个阶段的预测结果
    
    Args:
        start_idx: 起始样本索引
        num_samples: 要处理的样本数量
    """
    all_results = []
    
    for i in range(num_samples):
        sample_idx = start_idx + i
        if sample_idx >= len(question_image_data):
            print(f"已达到数据集末尾，共处理了{i}个样本")
            break
            
        print("\n" + "="*50)
        print(f"处理样本 {sample_idx} (序号 {i+1}/{num_samples})")
        print("="*50)

        # 获取问题和图像ID
        question = question_image_data[sample_idx]['question']
        image_id = question_image_data[sample_idx]['image_id']
        print(f"问题: {question}")
        print(f"图像ID: {image_id}")

        # 读取图像
        image_path = os.path.join(COMBINED_IMAGE_DIR, image_id)
        print(f"图像路径: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像，跳过此样本")
            # 记录错误信息
            error_result = {
                "sample_id": sample_idx,
                "image_id": image_id,
                "question": question,
                "error": "无法读取图像"
            }
            all_results.append(error_result)
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建单个样本的结果字典
        sample_result = {
            "sample_id": sample_idx,
            "image_id": image_id,
            "question": question,
            "stages": {}
        }

        # 阶段1: ViLT预测可能答案
        print("\n=== 阶段1: ViLT预测可能答案 ===")
        pred_answers = inference_vilt(input_image=image,
                                      input_text=question,
                                      print_preds=True)  # 设置为True输出预测详情
        
        # 记录阶段1结果
        sample_result["stages"]["stage1_vilt"] = {
            "predicted_answers": pred_answers
        }
        print(f"ViLT预测答案: {pred_answers}")

        # 阶段2: 视觉定位
        print("\n=== 阶段2: 视觉定位 ===")
        predicted_grounding_masks = []
        mask_coverage_data = []
        
        try:
            image_pil = Image.open(image_path)
            for j, ans in enumerate(pred_answers):
                # 构建输入文本
                input_text = question + " answer:" + ans
                print(f"视觉定位输入 ({j+1}/{len(pred_answers)}): {input_text}")
                
                # 执行视觉定位
                pred_overlayed, pred_mask = visual_grounding(image=image_pil, text=input_text)
                predicted_grounding_masks.append(pred_mask)
                
                # 计算掩码覆盖区域的比例
                mask_coverage = np.mean(pred_mask)
                mask_coverage_data.append({
                    "answer": ans,
                    "coverage": float(mask_coverage)
                })
                print(f"答案 '{ans}' 的掩码覆盖率: {mask_coverage:.4f}")
        except Exception as e:
            print(f"视觉定位过程中出错: {e}")
            sample_result["stages"]["stage2_grounding"] = {
                "error": str(e)
            }
            sample_result["stages"]["stage3_final"] = {
                "error": "由于阶段2错误，无法执行阶段3"
            }
            all_results.append(sample_result)
            continue
        
        # 记录阶段2结果
        sample_result["stages"]["stage2_grounding"] = {
            "answers": pred_answers,
            "mask_coverage": mask_coverage_data
        }

        # 阶段3: 单/多目标判断
        print("\n=== 阶段3: 单/多目标判断 ===")
        predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
        is_single = 1 if predicted_label == 'single' else 0
        
        # 记录阶段3结果
        sample_result["stages"]["stage3_final"] = {
            "predicted_label": predicted_label,
            "is_single": is_single
        }
        print(f"最终预测标签: {predicted_label} (is_single={is_single})")
        
        # 将样本结果添加到总结果列表
        all_results.append(sample_result)
        
        # 实时保存结果
        with open(all_stages_output_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n样本 {sample_idx} 处理完成，已保存结果")
    
    print(f"\n所有样本处理完成，共处理了 {len(all_results)} 个样本")
    print(f"结果已保存到 {all_stages_output_file}")
    
    return all_results

# 批量处理所有样本
def process_all_samples():
    """
    处理所有样本，输出每个阶段的预测结果
    """
    stage1_results = []  # ViLT预测结果
    stage2_results = []  # 视觉定位结果
    stage3_results = []  # 最终单/多目标判断结果
    submission_results = []  # 提交格式结果
    
    maximum_length = 35  # 问题最大长度
    
    print(f"开始处理所有样本，共 {len(question_image_data)} 个...")
    
    for i, item in enumerate(tqdm(question_image_data)):
        image_id = item['image_id']
        raw_question = item['question']
        
        # 处理问题长度
        number_of_chars = len(raw_question)
        if number_of_chars > maximum_length:
            question = raw_question[:maximum_length]
        else:
            question = raw_question
            
        # 图像路径
        image_path = os.path.join(COMBINED_IMAGE_DIR, image_id)
        print(f"\n正在处理样本 {i+1}/{len(question_image_data)}")
        print(f"图像ID: {image_id}")
        print(f"问题: {question}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像，跳过此样本")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 阶段1: ViLT预测可能答案
        print("阶段1: ViLT预测可能答案")
        pred_answers = inference_vilt(input_image=image,
                                     input_text=question,
                                     print_preds=False)
        
        # 记录阶段1结果
        stage1_result = {
            "sample_id": i,
            "image_id": image_id,
            "question": question,
            "predicted_answers": pred_answers
        }
        stage1_results.append(stage1_result)
        print(f"ViLT预测答案: {pred_answers}")
        
        # 阶段2: 视觉定位
        print("阶段2: 视觉定位")
        predicted_grounding_masks = []
        mask_coverage_data = []
        
        try:
            image_pil = Image.open(image_path)
            for j, ans in enumerate(pred_answers):
                # 构建输入文本
                input_text = question + " answer:" + ans
                
                # 执行视觉定位
                pred_overlayed, pred_mask = visual_grounding(image=image_pil, text=input_text)
                predicted_grounding_masks.append(pred_mask)
                
                # 计算掩码覆盖区域的比例
                mask_coverage = np.mean(pred_mask)
                mask_coverage_data.append({
                    "answer": ans,
                    "coverage": float(mask_coverage)
                })
                print(f"答案 '{ans}' 的掩码覆盖率: {mask_coverage:.4f}")
        except Exception as e:
            print(f"视觉定位过程中出错: {e}")
            continue
        
        # 记录阶段2结果
        stage2_result = {
            "sample_id": i,
            "image_id": image_id,
            "question": question,
            "answers": pred_answers,
            "mask_coverage": mask_coverage_data
        }
        stage2_results.append(stage2_result)
        
        # 阶段3: 单/多目标判断
        print("阶段3: 单/多目标判断")
        predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
        is_single = 1 if predicted_label == 'single' else 0
        
        # 记录阶段3结果
        stage3_result = {
            "sample_id": i,
            "image_id": image_id,
            "question": question,
            "predicted_label": predicted_label,
            "is_single": is_single,
            "mask_coverage_data": mask_coverage_data
        }
        stage3_results.append(stage3_result)
        
        # 创建提交结果
        submission_entry = {
            'question_id': image_id,
            'single_grounding': is_single
        }
        submission_results.append(submission_entry)
        
        print(f"已处理 {i+1}/{len(question_image_data)} 个样本，预测结果: {predicted_label}")
        
        # 每处理50个样本保存一次中间结果
        if (i + 1) % 50 == 0 or (i + 1) == len(question_image_data):
            # 保存各阶段结果到文件
            with open(stage1_output_file, 'w') as f:
                json.dump(stage1_results, f, indent=2, ensure_ascii=False)
            
            with open(stage2_output_file, 'w') as f:
                json.dump(stage2_results, f, indent=2, ensure_ascii=False)
            
            with open(stage3_output_file, 'w') as f:
                json.dump(stage3_results, f, indent=2, ensure_ascii=False)
                
            # 保存提交结果
            with open('/root/autodl-tmp/output/submission_interim_yuanshi.json', 'w') as f:
                json.dump(submission_results, f, indent=2)
                
            print(f"已保存中间结果，当前进度: {i+1}/{len(question_image_data)}")

    # 保存最终结果
    with open(stage1_output_file, 'w') as f:
        json.dump(stage1_results, f, indent=2, ensure_ascii=False)
    
    with open(stage2_output_file, 'w') as f:
        json.dump(stage2_results, f, indent=2, ensure_ascii=False)

    with open(stage3_output_file, 'w') as f:
        json.dump(stage3_results, f, indent=2, ensure_ascii=False)
                  
    # # 保存最终结果
    # with open(vilt_predictions_file, 'w') as f:
    #     json.dump(all_vilt_predictions, f)
    
    # with open(grounding_predictions_file, 'w') as f:
    #     json.dump(all_grounding_predictions, f)
    
    # with open(final_predictions_file, 'w') as f:
    #     json.dump(all_final_predictions, f)
    
    # 保存提交结果
    with open('/root/autodl-tmp/output/submission_yuanshi.json', 'w') as f:
        json.dump(submission_results, f)
    
    print(f"所有结果已保存")

# 调用批量处理函数
print("开始批量处理所有样本...")
process_all_samples()
print("处理完成!")