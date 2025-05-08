from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering

import numpy as np
import cv2
import matplotlib.pyplot as plt

finetune_folder = '/root/autodl-tmp/output/models/custom_vqa_vilt-b32-finetuned-vqa'
finetune_folder = os.path.join('/root/autodl-tmp/output/models/custom_vqa_vilt-b32-finetuned-vqa')


colors = [(255,0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 255), (128, 0, 0), (0, 128, 0),
          (0, 0, 128)]
thicknesses = [3,3,3, 3, 3, 3, 3, 3, 3, 3]
# Helper function
def is_single_groundings(predicted_grounding_masks: List[np.ndarray], threshold=0.3):
    if len(predicted_grounding_masks) == 0:
        return False
    reference_mask = predicted_grounding_masks[0]
    for mask in predicted_grounding_masks[1:]:
        if mask.shape != reference_mask.shape:
            return False
        intersection = np.logical_and(mask, reference_mask)
        union = np.logical_or(mask, reference_mask)
        iou = np.sum(intersection) / np.sum(union)
        if iou < threshold:
            return False
    return True

def draw_boundaries(image, mask, color=(55, 0, 200)):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaries_image = cv2.drawContours(image.copy(), contours, -1, color, 2)
    return boundaries_image

def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def calculate_iou_mask(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    iou = intersection_count / union_count
    return iou


def plot_grid(result_images, num_rows, num_cols=1):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))
    for i, ax in enumerate(axes.flat):
        fig_header = "Prediction"
        ax.imshow(result_images[i])
        ax.set_title(fig_header, fontsize=10, pad=5)
        ax.axis('off')
    fig.tight_layout()
    figure_path = './eval_samples.png'
    plt.savefig(figure_path)


class VQADataset(torch.utils.data.Dataset):
    """VQA dataset."""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        data = self.data[idx]

        image_id = str(data["image_id"])
        if "jpg" not in image_id:
            image_id = image_id.zfill(12) + '.jpg'
        img_path = os.path.join(vqa_datatrain_image_dir, image_id)
        image = Image.open(img_path)
        if image.mode == "L":
            # If image is in grayscale, convert to RGB
            image = image.convert("RGB")
        text = data['question']
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = []
        for ans in data["answers"]:
            labels.append(config.label2id[ans])
        scores = [0.8] * len(labels)

        # labels = annotation['labels']
        # scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))

        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding


def handle_dataset(df):
    print("Handle dataset .... ")
    label_set = set()
    for ans_list in df["answers"]:
        for ans in ans_list:
            label_set.add(ans)
    unique_labels = list(label_set)
    config.label2id = {label: idx for idx, label in enumerate(unique_labels)}
    config.id2label = {idx: label for label, idx in config.label2id.items()}


def generate_dataset(file_path) -> VQADataset:
    dataframe = pd.read_json(file_path)

    dataframe_dict = dataframe.to_dict(orient='records')
    handle_dataset(dataframe)

    dataset = VQADataset(data=dataframe_dict,
                         processor=processor)

    return dataset


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch


def train_model(model, dataset, num_epochs=50):
    train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs;
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss = loss.item()
            print("Loss:", epoch_loss)
            epoch_loss += epoch_loss
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_dataloader)
        # Store loss value
        losses.append(epoch_loss)

        # Plot and save the loss graph
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_graph.png')
    print("Total missing words", len(missing_words_in_vocab))

    # Save the model weights
    model.save_pretrained(finetune_folder)
    # Save the feature extractor
    processor.save_pretrained(finetune_folder)

def get_model():
    print("Get model from pretrained: ", model_folder)
    model = ViltForQuestionAnswering.from_pretrained(
        pretrained_model_name_or_path=model_folder,
        id2label=config.id2label,
        label2id=config.label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model.to(device)

    return model
#     dataset_vqa = generate_dataset(vqa_datatrain_annotation_path)
#     my_model = get_model()
#     train_model(my_model, dataset_vqa, 10)

# def inference_vilt(input_image, input_text, threshold=0.1, print_preds=True):
#     print("Inference finetune.....")
#     fintune_processor = ViltProcessor.from_pretrained(finetune_folder)
#     model = ViltForQuestionAnswering.from_pretrained(finetune_folder)
#
#     # prepare inputs
#     encoding = fintune_processor(input_image, input_text, return_tensors="pt")
#
#     # forward pass
#     outputs = model(**encoding)
#     logits = outputs.logits
#     predicted_classes = torch.sigmoid(logits)
#     probs, classes = torch.topk(predicted_classes, 5)
#
#     results = []
#     for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
#         if prob >= threshold:
#             results.append(model.config.id2label[class_idx])
#         if print_preds:
#             print(prob, model.config.id2label[class_idx])
#
#     if not results:
#         class_idx = logits.argmax(-1).item()
#         results.append(model.config.id2label[class_idx])
#     return results

def inference_vilt(input_image, input_text, threshold=0.01, print_preds=True):
    # print("Inference finetune.....")
    fintune_processor = ViltProcessor.from_pretrained(finetune_folder)
    model = ViltForQuestionAnswering.from_pretrained(finetune_folder)
    # prepare inputs
    encoding = fintune_processor(input_image, input_text, return_tensors="pt")
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_classes = torch.sigmoid(logits)
    probs, classes = torch.topk(predicted_classes, 5)
    results = []
    num_detected = 0
    for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
        if prob >= threshold:
            results.append(model.config.id2label[class_idx])
            num_detected += 1
            if num_detected >= 3:
                break
    if not results:
        try:
            class_idx = logits.argmax(-2).item()
        except:
            class_idx = logits.argmax(-1).item()
        results.append(model.config.id2label[class_idx])
    return results

from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import cv2
import matplotlib.pyplot as plt

finetune_folder = '/root/autodl-tmp/models/blip2-flan-t5-xl'
finetune_folder = os.path.join('/root/autodl-tmp/models/blip2-flan-t5-xl')

colors = [(255,0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 255), (128, 0, 0), (0, 128, 0),
          (0, 0, 128)]
thicknesses = [3,3,3, 3, 3, 3, 3, 3, 3, 3]

# Helper function
def is_single_groundings(predicted_grounding_masks: List[np.ndarray], threshold=0.2):
    if len(predicted_grounding_masks) == 0:
        return False
    reference_mask = predicted_grounding_masks[0]
    for mask in predicted_grounding_masks[1:]:
        if mask.shape != reference_mask.shape:
            return False
        intersection = np.logical_and(mask, reference_mask)
        union = np.logical_or(mask, reference_mask)
        iou = np.sum(intersection) / np.sum(union)
        if iou < threshold:
            return False
    return True

def draw_boundaries(image, mask, color=(55, 0, 200)):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaries_image = cv2.drawContours(image.copy(), contours, -1, color, 2)
    return boundaries_image

def coordinates_to_mask(coordinates, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def calculate_iou_mask(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    iou = intersection_count / union_count
    return iou

def plot_grid(result_images, num_rows, num_cols=1):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))
    for i, ax in enumerate(axes.flat):
        fig_header = "Prediction"
        ax.imshow(result_images[i])
        ax.set_title(fig_header, fontsize=10, pad=5)
        ax.axis('off')
    fig.tight_layout()
    figure_path = './eval_samples.png'
    plt.savefig(figure_path)

class VQADataset(torch.utils.data.Dataset):
    """VQA dataset."""

    def __init__(self, data, processor, config):
        self.data = data
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        data = self.data[idx]

        image_id = str(data["image_id"])
        if "jpg" not in image_id:
            image_id = image_id.zfill(12) + '.jpg'
        img_path = os.path.join(vqa_datatrain_image_dir, image_id)
        image = Image.open(img_path)
        if image.mode == "L":
            # If image is in grayscale, convert to RGB
            image = image.convert("RGB")
        
        text = data['question']
        
        # Process inputs for BLIP
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
            
        # Add labels
        answers = data["answers"]
        # Convert answers to BLIP format
        # For BLIP, we'll use the first answer as the primary answer
        if len(answers) > 0:
            encoding["answer"] = answers[0]
            # We'll also store all possible answers for evaluation
            encoding["all_answers"] = answers
        else:
            encoding["answer"] = ""
            encoding["all_answers"] = []

        return encoding

def handle_dataset(df, config):
    print("Handle dataset .... ")
    label_set = set()
    for ans_list in df["answers"]:
        for ans in ans_list:
            label_set.add(ans)
    unique_labels = list(label_set)
    config.label2id = {label: idx for idx, label in enumerate(unique_labels)}
    config.id2label = {idx: label for label, idx in config.label2id.items()}

def generate_dataset(file_path, processor, config) -> VQADataset:
    dataframe = pd.read_json(file_path)
    dataframe_dict = dataframe.to_dict(orient='records')
    handle_dataset(dataframe, config)

    dataset = VQADataset(
        data=dataframe_dict,
        processor=processor,
        config=config
    )

    return dataset

def collate_fn(batch):
    # BLIP uses different input structure compared to ViLT
    input_ids = [item['input_ids'] for item in batch if 'input_ids' in item]
    pixel_values = [item['pixel_values'] for item in batch if 'pixel_values' in item]
    attention_mask = [item['attention_mask'] for item in batch if 'attention_mask' in item]
    answers = [item['answer'] for item in batch if 'answer' in item]
    
    # Create batch dictionary
    batch_dict = {}
    
    if input_ids:
        batch_dict['input_ids'] = torch.stack(input_ids)
    if attention_mask:
        batch_dict['attention_mask'] = torch.stack(attention_mask)
    if pixel_values:
        batch_dict['pixel_values'] = torch.stack(pixel_values)
    
    # For BLIP, we need to format the answers differently than ViLT
    if answers:
        batch_dict['labels'] = answers
    
    return batch_dict

def train_model(model, dataset, config, num_epochs=50):
    train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # BLIP expects different inputs
            # Check if 'labels' is in batch and convert to proper format if needed
            if 'labels' in batch:
                labels = batch.pop('labels')
                # Forward pass with BLIP
                outputs = model(
                    input_ids=batch.get('input_ids'),
                    pixel_values=batch.get('pixel_values'),
                    attention_mask=batch.get('attention_mask'),
                    labels=labels
                )
            else:
                # Forward pass without labels
                outputs = model(**batch)
            
            loss = outputs.loss
            current_loss = loss.item()
            print("Loss:", current_loss)
            epoch_loss += current_loss
            
            loss.backward()
            optimizer.step()
            
        epoch_loss /= len(train_dataloader)
        losses.append(epoch_loss)
        
        # Plot and save the loss graph
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_graph.png')
    
    print("Training complete")
    
    # Save the model and processor
    model.save_pretrained(finetune_folder)
    processor.save_pretrained(finetune_folder)

def get_model(model_folder, config):
    print("Get model from pretrained: ", model_folder)
    # Initialize BLIP model for question answering
    model = BlipForQuestionAnswering.from_pretrained(
        pretrained_model_name_or_path=model_folder,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    
    # No need to explicitly set id2label and label2id for BLIP
    # but we can use them for inference later
    model.config.id2label = config.id2label
    model.config.label2id = config.label2id
    
    return model

# def inference_blip(input_image, input_text, threshold=0.01, print_preds=True):
def inference_blip(input_image, input_text, threshold=0.3, print_preds=True):
    print("Performing BLIP inference...BLIP推理中")
    # Load the fine-tuned processor and model
    processor = BlipProcessor.from_pretrained(finetune_folder)
    model = BlipForQuestionAnswering.from_pretrained(finetune_folder)
    
    # Prepare inputs for BLIP
    inputs = processor(input_image, input_text, return_tensors="pt")
    
    # Forward pass
    outputs = model.generate(**inputs)
    
    # Decode the output
    generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    # For compatibility with the original function, we return the answer in a list
    results = [generated_answer]
    
    if print_preds:
        print(f"Question: {input_text}")
        print(f"Answer: {generated_answer}")
    
    return results

# # To use the code:
# # 1. Create config object
# class Config:
#     def __init__(self):
#         self.id2label = {}
#         self.label2id = {}

# # 2. Initialize processor and model
# # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
# # config = Config()
# # model_folder = "Salesforce/blip-vqa-base"
# # model = get_model(model_folder, config)

# # 3. Load and process dataset
# # dataset_vqa = generate_dataset(vqa_datatrain_annotation_path, processor, config)

# # 4. Train the model
# # train_model(model, dataset_vqa, config, num_epochs=10)

# # 5. Use for inference
# # results = inference_blip(image, question)