import os
import json
import random
import shutil
from tqdm import tqdm

# This script is used to split the VizWiz dataset into training and validation sets
# It divides the dataset into 80% training images and 20% validation images

def split_vizwiz_dataset(
    image_dir,           # Path to the original image directory
    annotation_file,     # Path to the annotation JSON file
    train_image_dir,     # Path to the directory where training images will be stored
    val_image_dir,       # Path to the directory where validation images will be stored
    train_anno_file,     # Path to the output file for training annotations
    val_anno_file,       # Path to the output file for validation annotations
    val_ratio=0.2,       # Proportion of the dataset to be used for validation (default 20%)
    random_seed=42       # Seed for random number generator (default 42)
):
    # Set the random seed for reproducibility
    random.seed(random_seed)
    
    # Ensure that output directories exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    
    # Read the annotation file
    print(f"Reading annotation file {annotation_file}...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Organize annotations by image ID
    anno_by_image = {}
    for anno in annotations:
        image_id = anno['image_id']
        anno_by_image[image_id] = anno
    
    # Get the list of all image IDs
    all_image_ids = list(anno_by_image.keys())
    total_images = len(all_image_ids)
    
    # Shuffle the image IDs randomly
    random.shuffle(all_image_ids)
    
    # Calculate the split point based on the validation ratio
    val_size = int(total_images * val_ratio)
    train_size = total_images - val_size
    
    # Split the image IDs into training and validation sets
    train_image_ids = all_image_ids[:train_size]
    val_image_ids = all_image_ids[train_size:]
    
    print(f"Total number of images: {total_images}")
    print(f"Number of training images: {len(train_image_ids)} ({100 * len(train_image_ids) / total_images:.2f}%)")
    print(f"Number of validation images: {len(val_image_ids)} ({100 * len(val_image_ids) / total_images:.2f}%)")
    
    # Initialize lists to store annotations for training and validation sets
    train_annotations = []
    val_annotations = []
    
    # Process the training set images and annotations
    print("Processing training set...")
    for img_id in tqdm(train_image_ids):
        # Copy the image to the training directory
        src_path = os.path.join(image_dir, img_id)
        dst_path = os.path.join(train_image_dir, img_id)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image {img_id} not found")
            
        # Add the corresponding annotation to the training annotations list
        train_annotations.append(anno_by_image[img_id])
    
    # Process the validation set images and annotations
    print("Processing validation set...")
    for img_id in tqdm(val_image_ids):
        # Copy the image to the validation directory
        src_path = os.path.join(image_dir, img_id)
        dst_path = os.path.join(val_image_dir, img_id)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image {img_id} not found")
            
        # Add the corresponding annotation to the validation annotations list
        val_annotations.append(anno_by_image[img_id])
    
    # Save the annotations for the training set
    print(f"Saving training annotations to {train_anno_file}...")
    with open(train_anno_file, 'w') as f:
        json.dump(train_annotations, f)
    
    # Save the annotations for the validation set
    print(f"Saving validation annotations to {val_anno_file}...")
    with open(val_anno_file, 'w') as f:
        json.dump(val_annotations, f)
    
    print("Dataset split complete!")

# Example usage
if __name__ == "__main__":
    # Modify these paths according to your actual directories
    split_vizwiz_dataset(
        image_dir="/root/autodl-tmp/dataset/train", # Path to original image directory
        annotation_file="/root/autodl-tmp/dataset/VizWiz_train.json", # Path to annotation JSON file
        train_image_dir="./split/vizwiz_train_images",    # Path to output training image directory
        val_image_dir="./split/vizwiz_val_images",        # Path to output validation image directory
        train_anno_file="./split/vizwiz_train_annotations.json", # Path to output training annotations
        val_anno_file="./split/vizwiz_val_annotations.json",     # Path to output validation annotations
        val_ratio=0.2,                            # Validation set ratio (default 20%)
        random_seed=42                            # Random seed (default 42)
    )
