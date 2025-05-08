from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import re
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import Blip2Processor, BlipForQuestionAnswering, GPT2Tokenizer, BlipImageProcessor, T5Tokenizer
from transformers import Blip2Config, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util 
from peft import PeftModel
from inference_utils import *

# Set environment variables
os.environ["TRANSFORMERS_NO_COMPILER"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add custom path for polygon-transformer demo and fairseq
sys.path.append('/root/autodl-tmp/BLaVe-CoT/VQA-init/src/polygon-transformer')
from demo import visual_grounding
sys.path.append('/root/autodl-tmp/BLaVe-CoT/VQA-init/src/polygon-transformer/fairseq')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set data paths
vizwiz_data_base_path = '/root/autodl-tmp/BLaVe-CoT/VQA-init/dataset'
viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'VizWiz-VQA-test')
viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz-VQA-test.json')

# Model folder paths
model_folder = os.path.join("/root/autodl-tmp/BLaVe-CoT/models", "blip2-opt-2.7b")

# Load default configuration and modify vision model settings
config = Blip2Config.from_pretrained(model_folder)
config.vision_config.hidden_size = 1408
config.vision_config.intermediate_size = 6144

# Load the base model and move it to GPU
base_model = Blip2ForConditionalGeneration.from_pretrained(
    model_folder,
    config=config
).to(device)

print(f"Loading Blip2ForConditionalGeneration from: {model_folder}")

# Load fine-tuned model using LoRA
model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/BLaVe-CoT/models/blip2_vqa_finetuned_epoch_40")
print(f"Loading model from: {model_folder}")

# Load image processor and tokenizer
image_processor = BlipImageProcessor.from_pretrained(model_folder)
tokenizer = GPT2Tokenizer.from_pretrained("/root/autodl-tmp/BLaVe-CoT/models/blip2-opt-2.7b")

# Combine image processor and tokenizer into a processor
processor = Blip2Processor(image_processor, tokenizer)

# Load the SentenceTransformer model for semantic similarity
print("Loading all-MiniLM-L6-v2 model for semantic similarity...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the VizWiz dataset
COMBINED_IMAGE_DIR = '/root/autodl-tmp/BLaVe-CoT/VQA-init/dataset/VizWiz-VQA-test'
question_image_data = json.load(open('/root/autodl-tmp/BLaVe-CoT/VQA-init/dataset/VizWiz-VQA-test.json'))

# Extract all questions from the dataset
questions = [item['question'] for item in question_image_data]
print(f"Total number of questions: {len(questions)}")

def simplify_question(input_text):
    """
    Simplify the question text by removing redundant greetings and polite phrases.
    """
    # Remove greetings and polite phrases
    input_text = re.sub(r"^(Hi|Hello|Hey|Good morning|Good evening)[^a-zA-Z0-9]*", "", input_text)
    input_text = re.sub(r"(please|thank you|thanks)[^a-zA-Z0-9]*", "", input_text)
    input_text = re.sub(r"\s{2,}", " ", input_text).strip()  # Remove extra spaces
    
    # Return the simplified question
    return input_text

def filter_and_weight_answers(pred_answers, is_common_answer, answer_confidences):
    """
    Filters and weights answers, processing useful answers in cases where some answers are more meaningful.
    
    Args:
        pred_answers: List of predicted answers.
        is_common_answer: Boolean list indicating which answers are common.
        answer_confidences: List of confidence scores for each answer.
    
    Returns:
        filtered_answers: Filtered list of answers.
        filtered_is_common: Filtered list of common answer flags.
        filtered_confidences: Filtered list of confidence scores.
        has_meaningful_answers: Flag indicating whether there are meaningful answers.
        is_numeric_question: Flag indicating whether the question is numeric (including time-related).
    """
    # Define meaningless terms
    meaningless_terms = ["strutconnector", "strut", "connector", "ve", "f", "court"]
    common_terms = ["unknown", "cannot tell", "can't tell", "yes", "no"]
    
    # Check if the question is numeric (including time-related)
    is_numeric_question = False
    numeric_pattern = re.compile(r'^\d{1,2}[:\.]\d{1,2}$|^\d{1,2}(am|pm)$|^\d+$|^\d+\.\d+$')
    numeric_answers_count = sum(1 for ans in pred_answers if numeric_pattern.match(ans.lower().strip()))
    
    if numeric_answers_count > 0:
        is_numeric_question = True
    
    filtered_answers = []
    filtered_is_common = []
    filtered_confidences = []
    
    # First round: collect all meaningful answers
    meaningful_answers = []
    meaningful_is_common = []
    meaningful_confidences = []
    
    for i, answer in enumerate(pred_answers):
        answer_lower = answer.lower().strip()
        
        # Check if it contains meaningless terms
        contains_meaningless = any(term in answer_lower for term in meaningless_terms)
        if contains_meaningless:
            # Try to clean the answer by removing meaningless terms
            cleaned_answer = answer_lower
            for term in meaningless_terms:
                cleaned_answer = cleaned_answer.replace(term, "").strip()
            
            if cleaned_answer and cleaned_answer not in common_terms:
                meaningful_answers.append(cleaned_answer)
                meaningful_is_common.append(False)
                meaningful_confidences.append(answer_confidences[i] * 0.8)
            continue
        
        # Check if it's a common term
        if answer_lower in common_terms:
            continue
        else:
            # Add meaningful answer
            meaningful_answers.append(answer)
            meaningful_is_common.append(is_common_answer[i])
            meaningful_confidences.append(answer_confidences[i])
    
    # If meaningful answers exist, use them
    if meaningful_answers:
        filtered_answers = meaningful_answers
        filtered_is_common = meaningful_is_common
        filtered_confidences = meaningful_confidences
        has_meaningful_answers = True
    else:
        # If no meaningful answers, consider common answers
        for i, answer in enumerate(pred_answers):
            answer_lower = answer.lower().strip()
            if answer_lower in common_terms:
                filtered_answers.append(answer)
                filtered_is_common.append(True)
                filtered_confidences.append(answer_confidences[i] * 0.5)
        
        if not filtered_answers:
            filtered_answers = [" "]
            filtered_is_common = [True]
            filtered_confidences = [0.5]
            has_meaningful_answers = False
        else:
            has_meaningful_answers = False
    
    return filtered_answers, filtered_is_common, filtered_confidences, has_meaningful_answers, is_numeric_question

def cot_verification(answers, threshold_high=0.7, threshold_low=0.2):
    """
    Chain of Thought verification for answer validation.
    
    Args:
        answers: List of model-generated answers.
        threshold_high: High similarity threshold.
        threshold_low: Low similarity threshold.
    
    Returns:
        is_single: Single/multiple target determination based on semantic similarity (1=single, 0=multiple).
        avg_similarity: Average semantic similarity between answers.
    """
    if len(answers) < 2:
        return 0, 0.0
    
    embeddings = similarity_model.encode(answers)
    similarities = []
    count = 0
    
    for i in range(len(answers)):
        for j in range(i+1, len(answers)):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            similarities.append(similarity)
            if similarity > threshold_high:
                count += 1
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    is_single = 1 if count > 0 else 0
    
    return is_single, avg_similarity

def generate_quality_answers(input_image, input_text, similarity_model, min_answers=3, max_answers=3):
    """
    Generate high-quality answers and evaluate their quality.
    
    Args:
        input_image: Input image.
        input_text: Input question.
        similarity_model: Semantic similarity model.
        min_answers: Minimum number of answers.
        max_answers: Maximum number of answers.
    
    Returns:
        answers: List of generated answers.
        is_common: Boolean list indicating which answers are common.
        confidences: Confidence scores for each answer.
    """
    # Ensure image is in PIL.Image format
    if not isinstance(input_image, Image.Image):
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
    
    # Prepare input
    inputs = processor(input_image, input_text, return_tensors="pt").to(device)
    
    # Generate answers
    outputs = model.generate(**inputs, max_length=20)
    generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    answers = []
    is_common = []
    confidences = []
    
    if generated_answer.strip():
        answers.append(generated_answer)
        is_common.append(False)
        confidences.append(1.0)
    
    if len(answers) < max_answers:
        try:
            beam_outputs = model.generate(
                **inputs, 
                max_length=20,
                num_beams=5,
                num_return_sequences=min(5, 5),
                early_stopping=True
            )
            
            for beam_output in beam_outputs:
                answer = processor.decode(beam_output, skip_special_tokens=True)
                if answer not in answers and answer.strip():
                    answers.append(answer)
                    is_common.append(False)
                    confidences.append(0.9)
                    if len(answers) >= max_answers:
                        break
        except Exception as e:
            print(f"Error generating multiple candidate answers: {e}")
    
    # If answers are insufficient, consider adding common answers
    if len(answers) < min_answers:
        common_answers = ["yes", "no", "unknown", "cannot tell"]
        
        if answers and similarity_model is not None:
            try:
                existing_embeddings = similarity_model.encode(answers)
                avg_embedding = np.mean(existing_embeddings, axis=0)
                common_embeddings = similarity_model.encode(common_answers)
                
                similarities = util.pytorch_cos_sim(avg_embedding, common_embeddings)[0].numpy()
                sorted_indices = np.argsort(-similarities)
                
                for idx in sorted_indices:
                    if common_answers[idx] not in answers:
                        answers.append(common_answers[idx])
                        is_common.append(True)
                        confidences.append(float(similarities[idx]))
                        if len(answers) >= min_answers:
                            break
            except Exception as e:
                print(f"Error selecting similar common answers: {e}")
                if len(answers) < min_answers and "unknown" not in answers:
                    answers.append("unknown")
                    is_common.append(True)
                    confidences.append(0.5)
        else:
            if "unknown" not in answers:
                answers.append("unknown")
                is_common.append(True)
                confidences.append(0.5)
    
    return answers, is_common, confidences

def process_all_samples():
    """
    Process all samples and output the final submission result.
    """
    submission_results = []  # Final submission results
    
    maximum_length = 150  # Max question length
    
    print(f"Starting processing of all samples, total {len(question_image_data)} samples...")
    
    for i, item in enumerate(tqdm(question_image_data)): 
        image_id = item['image_id']
        raw_question = item['question']
        
        # Handle question length
        number_of_chars = len(raw_question)
        if number_of_chars > maximum_length:
            question = raw_question[:maximum_length]
        else:
            question = raw_question
        
        # Step 1: Simplify question text
        question = simplify_question(question)
        
        # Image path based on image_id
        if image_id.isdigit():
            image_filename = f"COCO_train2014_{int(image_id):012d}.jpg"
        else:
            image_filename = image_id  # Directly use image_id as filename
        
        image_path = os.path.join(COMBINED_IMAGE_DIR, image_filename)
        
        print(f"\nProcessing sample {i+1}/{len(question_image_data)}")
        print(f"Image ID: {image_id}")
        print(f"Question: {question}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image, skipping this sample")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stage 1: BLIP prediction of potential answers
        print("Stage 1: BLIP Prediction")
        pred_answers, is_common_answer, answer_confidences = generate_quality_answers(
            input_image=image,
            input_text=question,
            similarity_model=similarity_model,
            min_answers=3,
            max_answers=3
        )

        # Filter and weight answers
        filtered_answers, filtered_is_common, filtered_confidences, has_meaningful_answers, is_numeric_question = filter_and_weight_answers(
            pred_answers, is_common_answer, answer_confidences
        )

        print(f"Filtered answers: {filtered_answers}")
        
        # Stage 2: Visual grounding
        print("Stage 2: Visual Grounding")
        predicted_grounding_masks = []
        mask_coverage_data = []
        
        try:
            image_pil = Image.open(image_path)
            for j, ans in enumerate(filtered_answers):
                input_text = question + " answer:" + ans
                pred_overlayed, pred_mask = visual_grounding(image=image_pil, text=input_text)
                predicted_grounding_masks.append(pred_mask)
                mask_coverage = np.mean(pred_mask)
                mask_coverage_data.append({
                    "answer": ans,
                    "coverage": float(mask_coverage),
                    "is_common": is_common_answer[j],
                    "confidence": float(answer_confidences[j])
                })
                print(f"Answer '{ans}' mask coverage: {mask_coverage:.4f}")
        except Exception as e:
            print(f"Error during visual grounding: {e}")
            continue
        
        # Stage 3: Single/multiple target determination
        print("Stage 3: Single/Multiple Target Determination")

        # Initialize variables for stage 3
        mask_predicted_label = 'multiple'
        mask_is_single = 0
        cot_predicted_label = 'multiple'
        cot_is_single = 0
        avg_similarity = 0.0

        mask_predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
        mask_is_single = 1 if mask_predicted_label == 'single' else 0
        
        if is_numeric_question:
            final_is_single = mask_is_single
            final_predicted_label = mask_predicted_label
            decision_method = "numeric_question_mask_based"
            
            print(f"Numeric question detected, using mask-based result: {final_predicted_label}")
        elif not has_meaningful_answers:
            final_is_single = 1
            final_predicted_label = 'single'
            decision_method = "default_due_to_no_meaningful_answers"
            print(f"No meaningful answers, using default result: {final_predicted_label}")
        else:
            cot_is_single, avg_similarity = cot_verification(filtered_answers)
            cot_predicted_label = 'single' if cot_is_single == 1 else 'multiple'

            if len(filtered_answers) == 1 and not filtered_is_common[0]:
                final_is_single = 1
                final_predicted_label = 'single'
                decision_method = "single_meaningful_answer"
            else:
                if mask_is_single == cot_is_single:
                    final_is_single = mask_is_single
                    final_predicted_label = mask_predicted_label
                    decision_method = "agreement"
                else:
                    if avg_similarity < 0.2:
                        final_is_single = cot_is_single
                        final_predicted_label = cot_predicted_label
                        decision_method = "cot_override_low_similarity"
                    else:
                        final_is_single = mask_is_single
                        final_predicted_label = mask_predicted_label
                        decision_method = "mask_override_medium_similarity"
        
        submission_entry = {
            'question_id': image_id,
            'single_grounding': final_is_single
        }
        submission_results.append(submission_entry)
        
        print(f"Processed {i+1}/{len(question_image_data)} samples, prediction result: {final_is_single}")
    
    # Save only the final submission results
    final_output_dir = '/root/autodl-tmp/BLaVe-CoT/output'  # Set your path here
    with open(os.path.join(final_output_dir, 'submission_results.json'), 'w') as f:
        json.dump(submission_results, f, indent=2)
    
    print("All results have been saved!")

# Call the function to process all samples
print("Starting batch processing of all samples...")
process_all_samples()
print("Processing completed!")