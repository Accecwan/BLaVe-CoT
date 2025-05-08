# from typing import List
# import sys
# import os
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import json
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import ViltConfig
# from transformers import ViltProcessor
# from transformers import ViltForQuestionAnswering
# from inference_utils import *

# # sys.path.append('../polygon-transformer/')
# # sys.path.append('../polygon-transformer/fairseq')
# sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer')  # 添加 polygon-transformer 目录到模块搜索路径中
# sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer/fairseq')
# # sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'polygon-transformer'))

# # # 然后你就可以导入 demo 中的 visual_grounding 了
# from demo import visual_grounding
# # # from demo import visual_grounding

# vizwiz_data_base_path = '/root/autodl-tmp/dataset'

# viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
# viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')

# pretrained_model = 'vilt-b32-finetuned-vqa'
# model_folder = os.path.join('/root/autodl-tmp/models', pretrained_model)
# finetune_folder = '/root/autodl-tmp/output/models/custom_vqa_vilt-b32-finetuned-vqa'
# finetune_folder = os.path.join('/root/autodl-tmp/output/models/custom_vqa_vilt-b32-finetuned-vqa')


# processor = ViltProcessor.from_pretrained(model_folder)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# missing_words_in_vocab = []

# COMBINED_IMAGE_DIR = '/root/autodl-tmp/dataset/test'
# question_image_data = '/root/autodl-tmp/dataset/VizWiz_test.json'
# question_image_data = json.load(open(question_image_data))
# print(type(question_image_data))  # 查看 question_image_data 的类型
# print(question_image_data[0])  # 打印第一个元素，了解结构
# # print(question_image_data)        # 打印其内容，检查结构

# # print(len(question_image_data['questions']))
# # 提取所有问题到一个新的列表
# questions = [item['question'] for item in question_image_data]
# # 打印该列表的长度，类似于原始代码
# print(len(questions))  # 获取所有问题的数量




# # num = 111
# # for i in range(20):
# #     i = i + num
# #     print("=====Input information: {}".format(i))
# #     question = question_image_data['questions'][i]
# #     print(question)
# #     image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data['images'][i])
# #     image = cv2.imread(image_path)
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     plt.imshow(image)
# #     plt.show()
# #     print("Inference using finetune model")
# #     pred_answers: List[str] = inference_vilt(input_image=image,
# #                                              input_text=question,
# #                                              print_preds=False)
# #     predicted_grounding_masks: list = []
# #     image = Image.open(image_path)
# #     for i, ans in enumerate(pred_answers):
# #         # input
# #         input_text = question + " answer:" + ans
# #         # input_text = ans

# #         # input_text = question + " answer: sweet potato" + ans
# #         print(input_text)
# #         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
# #         plt.imshow(pred_mask)
# #         plt.show()
# #         predicted_grounding_masks.append(pred_mask)
# #         # save to mask
# #     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
# #     print("Final predicted label: ", predicted_label)
# #     # prediction
# #     break
# # print("Done")

# num = 111
# for i in range(20):
#     i = i + num
#     print("=====Input information: {}".format(i))

#     # 修改：直接访问字典中的 'question' 键
#     question = question_image_data[i]['question']
#     print(question)

#     # 使用正确的方式访问图片路径
#     image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data[i]['image_id'])
#     print(f"尝试读取图像: {image_path}")
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.show()

#     print("Inference using finetune model")
#     pred_answers: List[str] = inference_vilt(input_image=image,
#                                              input_text=question,
#                                              print_preds=False)
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)

#     for ans in pred_answers:
#         # 输入文本，构建新的输入文本
#         input_text = question + " answer:" + ans
#         print(input_text)

#         # 可视化结果
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         plt.imshow(pred_mask)
#         plt.show()
#         predicted_grounding_masks.append(pred_mask)

#     # 预测标签
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
#     print("Final predicted label: ", predicted_label)

#     # 终止循环，只运行一次
#     break

# print("Done")


# ## INFERENCE FOR SUBMISSION
# # naive_path = '/root/autodl-tmp/output/submission.json'
# # naive_data = json.load(open(naive_path))

# results = []
# # num = 694
# maximum_length = 35
# # for i in tqdm(range(len(question_image_data['questions']))):
# for i, item in enumerate(tqdm(question_image_data)):
#     image_id = item['image_id']
#     question = item['question']
#     # i = i + num
#     # raw_question = question_image_data['questions'][i]
#     raw_question = question_image_data[i]['question']
#     # print("=====Input information: {}".format(i))
#     # print(raw_question)
#     # print(question_image_data['images'][i])
#     number_of_chars = len(raw_question)
#     if number_of_chars > maximum_length:
#         question = raw_question[:maximum_length]
#         # print("Truncated question: ", question)
#     else:
#         question = raw_question
#         # print("Original question: ", question)
#     # image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data['images'][i])
#     image_path = os.path.join(COMBINED_IMAGE_DIR, question_image_data[i]['image_id'])
#     print(f"正在使用图像ID: {image_id}")
#     print(f"完整图像路径: {image_path}")

#     image = cv2.imread(image_path)  
#     # image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图像，跳过此样本")
#         continue

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # visualize
#     # plt.imshow(image)
#     # plt.show()
#     #
#     pred_answers: List[str] = inference_vilt(input_image=image,
#                                              input_text=question,
#                                              print_preds=False)
    
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)
#     for j, ans in enumerate(pred_answers):
#         # input
#         input_text = question + " answer:" + ans
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         predicted_grounding_masks.append(pred_mask)
#         # print(input_text)
#         # plt.imshow(pred_overlayed)
#         # plt.show()
#         # save to mask
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
    
#     # 创建结果字典
#     temp = {}
#     # temp['question_id'] = question_image_data['images'][i]
#     # temp['question_id'] = question_image_data[i]['images_id']
#     temp['question_id'] = image_id  # 使用 image_id 作为 question_id
#     temp['single_grounding'] = 1 if predicted_label == 'single' else 0
#     results.append(temp)
#     print(f"已处理 {i+1}/{len(question_image_data)} 个样本，预测结果: {predicted_label}")


#     # break
# # save to json
# # 保存为 JSON
# output_path = '/root/autodl-tmp/output/submission.json'  # 确保目录存在
# os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录（如果不存在）

# with open('submission.json', 'w') as f:
#     json.dump(results, f)
# print(f"结果已保存至: {output_path}")

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
# 将ViLT相关的导入替换为BLIP相关导入
from transformers import Blip2Processor, BlipForQuestionAnswering, T5Tokenizer, BlipImageProcessor
from transformers import Blip2Config, Blip2ForConditionalGeneration
from inference_utils import *

# 路径设置保持不变
sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer')
sys.path.append('/root/autodl-tmp/VQA-init/src/polygon-transformer/fairseq')
from demo import visual_grounding

vizwiz_data_base_path = '/root/dataset-split/split'
viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'vizwiz_val_images')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'vizwiz_val_annotations.json')

pretrained_model = 'blip2-flan-t5-xl'
model_folder = os.path.join('/root/autodl-tmp/models', pretrained_model)
finetune_folder = '/root/autodl-tmp/models/blip2-flan-t5-xl'

# 首先加载默认配置，然后修改视觉模型部分
config = Blip2Config.from_pretrained(model_folder)

# 修改配置中的视觉编码器隐藏维度
config.vision_config.hidden_size = 1408
config.vision_config.intermediate_size = 6144

# 使用修改后的配置创建模型
model = Blip2ForConditionalGeneration.from_pretrained(
    model_folder,
    config=config
).to('cuda')

# 使用BlipProcessor替代ViltProcessor
# processor = BlipProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = Blip2Processor.from_pretrained(model_folder)
# 分别加载图像处理器和tokenizer
image_processor = BlipImageProcessor.from_pretrained(model_folder)
tokenizer = T5Tokenizer.from_pretrained("/root/autodl-tmp/models/blip2-flan-t5-xl")  # 或者使用具体的T5模型名称

# 然后组合它们
processor = Blip2Processor(image_processor, tokenizer)


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
stage1_output_file = '/root/autodl-tmp/output/predictions/stage1_blip_results.json'  # 修改为BLIP
stage2_output_file = '/root/autodl-tmp/output/predictions/stage2_grounding_results.json'
stage3_output_file = '/root/autodl-tmp/output/predictions/stage3_final_results.json'
all_stages_output_file = '/root/autodl-tmp/output/predictions/all_stages_results.json'

import re
def simplify_question(input_text):
    """
    简化问题文本，去除冗余的问候语和礼貌用语
    """
    # 去除问候语和礼貌用语
    input_text = re.sub(r"^(Hi|Hello|Hey|Good morning|Good evening)[^a-zA-Z0-9]*", "", input_text)
    input_text = re.sub(r"(please|thank you|thanks)[^a-zA-Z0-9]*", "", input_text)
    input_text = re.sub(r"\s{2,}", " ", input_text).strip()  # 去除多余的空格
    
    # 返回简化后的问题
    return input_text

def optimize_color_question(input_text):
    """
    针对颜色问题进行优化，明确物体并统一问题格式
    """
    # 检查问题是否是关于颜色的（比如含有 "color"）
    if re.search(r"(color|colour|shade|hue|tone)", input_text, re.IGNORECASE):
        # 去除问题中的冗余部分，比如 "Can you tell me"
        input_text = re.sub(r"(Can you|Could you|Please|Would you|Tell me|I want to know|Is it possible)", "", input_text, flags=re.IGNORECASE)
        input_text = input_text.strip()  # 去除多余的空格
        
        # 如果问题是 "What color is this?"，增加物体描述
        if re.search(r"what color is this\?", input_text, re.IGNORECASE):
            # 这里可以根据实际情况对物体做补充，例如：自动识别“sofa”或“jacket”等
            input_text = "What color is this object?"
            
        # 如果问题是 "What is the color of this?"，增加物体描述
        elif re.search(r"what is the color of this\?", input_text, re.IGNORECASE):
            # 增加物体描述
            input_text = "What is the color of this object?"
        
    return input_text

# 添加BLIP推理函数
def inference_blip(input_image, input_text, threshold=0.01, print_preds=True):
    """
    使用BLIP模型进行问答推理
    
    Args:
        input_image: 输入图像（PIL.Image或numpy数组）
        input_text: 输入文本（问题）
        threshold: 答案的最小置信度阈值
        print_preds: 是否打印预测结果
    
    Returns:
        list: 预测的答案列表
    """
    # 加载BLIP模型和处理器
    # model = BlipForQuestionAnswering.from_pretrained(finetune_folder).to(device)
    
    # 确保图像是PIL.Image格式
    if not isinstance(input_image, Image.Image):
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
    
    # 准备输入
    inputs = processor(input_image, input_text, return_tensors="pt").to(device)
    
    # 生成答案
    outputs = model.generate(**inputs, max_length=20)
    
    # 解码输出
    generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 为了兼容原有函数返回格式，我们将答案放入列表中
    # 在实际应用中，BLIP可能会生成单个答案，而不是多个候选答案
    # 这里我们可以考虑返回不同的生成选项
    results = []
    
    # 如果生成的答案为空，尝试使用不同的方法或返回默认答案
    if not generated_answer.strip():
        results.append("unknown")
    else:
        results.append(generated_answer)
    
    # 可以尝试使用不同的生成参数获取多个答案
    if len(results) < 3:  # 确保至少有3个候选答案，与原始代码类似
        try:
            # 使用beam search生成多个候选答案
            beam_outputs = model.generate(
                **inputs, 
                max_length=20,
                num_beams=5,
                num_return_sequences=min(5, 5),
                early_stopping=True
            )
            
            for beam_output in beam_outputs:
                answer = processor.decode(beam_output, skip_special_tokens=True)
                if answer not in results and answer.strip():
                    results.append(answer)
                    if len(results) >= 3:
                        break
        except Exception as e:
            if print_preds:
                print(f"生成多个候选答案时出错: {e}")
    
    # 如果仍然没有足够的答案，添加一些常见的VQA答案
    common_answers = ["yes", "no", "unknown", "cannot tell", "1", "2", "3"]
    for ans in common_answers:
        if len(results) >= 3:
            break
        if ans not in results:
            results.append(ans)
    
    if print_preds:
        print(f"问题: {input_text}")
        print(f"BLIP预测答案: {results}")
    
    return results[:3]  # 最多返回3个答案，与原始代码保持一致


# 批量处理所有样本
def process_all_samples():
    """
    处理所有样本，输出每个阶段的预测结果
    """
    stage1_results = []  # BLIP预测结果
    stage2_results = []  # 视觉定位结果
    stage3_results = []  # 最终单/多目标判断结果
    submission_results = []  # 提交格式结果
    
    maximum_length = 50  # 问题最大长度
    
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
        
        # 1. 简化问题文本
        question = simplify_question(question)
        
        # 2. 针对颜色问题进行优化
        question = optimize_color_question(question)
        
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

        # 阶段1: BLIP预测可能答案
        print("阶段1: BLIP预测可能答案")
        pred_answers = inference_blip(input_image=image,
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
        print(f"BLIP预测答案: {pred_answers}")
                
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
            with open('/root/autodl-tmp/output/submission_interim.json', 'w') as f:
                json.dump(submission_results, f, indent=2)
                
            print(f"已保存中间结果，当前进度: {i+1}/{len(question_image_data)}")

    # 保存最终结果
    with open(stage1_output_file, 'w') as f:
        json.dump(stage1_results, f, indent=2, ensure_ascii=False)
    
    with open(stage2_output_file, 'w') as f:
        json.dump(stage2_results, f, indent=2, ensure_ascii=False)

    with open(stage3_output_file, 'w') as f:
        json.dump(stage3_results, f, indent=2, ensure_ascii=False)
    
    # 保存提交结果
    with open('/root/autodl-tmp/output/submission.json', 'w') as f:
        json.dump(submission_results, f)
    
    print(f"所有结果已保存")

# 调用批量处理函数
print("开始批量处理所有样本...")
process_all_samples()
print("处理完成!")