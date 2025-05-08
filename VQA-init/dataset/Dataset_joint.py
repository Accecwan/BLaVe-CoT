import json
import shutil
import os

# Merge the VQAv2 dataset and the VizWiz dataset into VizWiz-VQA-test

# Set paths
# json_file_path = '/root/autodl-tmp/dataset/VQA_test.json'
json_file_path = '/root/autodl-tmp/dataset/vizwiz_test.json'
# input_image_dir = '/root/autodl-tmp/VQAv2/train2014'
input_image_dir = '/root/autodl-tmp/dataset/vizwiz_test'
output_image_dir = '/root/autodl-tmp/VQA-init/dataset/VizWiz-VQA-test'

# Create output directory
os.makedirs(output_image_dir, exist_ok=True)

# Read the VQA_test.json file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract the image IDs involved
image_ids = [entry['image_id'] for entry in data]

# Loop through each image_id and copy the images to the target directory
for image_id in image_ids:
    # Build the image filename
    # image_filename = f'COCO_train2014_{int(image_id):012d}.jpg'
    image_filename = image_id
    input_image_path = os.path.join(input_image_dir, image_filename)
    
    # If the image file exists, copy it to the target directory
    if os.path.exists(input_image_path):
        output_image_path = os.path.join(output_image_dir, image_filename)
        shutil.copy(input_image_path, output_image_path)
        print(f"Image {image_filename} has been copied to {output_image_path}")
    else:
        print(f"Image {image_filename} does not exist in the input directory")

print("All images have been copied.")
