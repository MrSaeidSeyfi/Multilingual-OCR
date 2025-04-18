import json
import os
import shutil
from tqdm import tqdm
import random

def gather_all_data(input_dir, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    temp_img_dir = os.path.join(temp_dir, "images")
    temp_ann_dir = os.path.join(temp_dir, "annotations")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_ann_dir, exist_ok=True)
    
    train_json = os.path.join(input_dir, "labels", "publaynet", "train.json")
    val_json = os.path.join(input_dir, "labels", "publaynet", "val.json")
    
    all_images = []
    all_annotations = []
    existing_images = set()
    
    with open(train_json, 'r') as f:
        train_data = json.load(f)
        all_images.extend(train_data['images'])
        all_annotations.extend(train_data['annotations'])
    
    with open(val_json, 'r') as f:
        val_data = json.load(f)
        all_images.extend(val_data['images'])
        all_annotations.extend(val_data['annotations'])
    
    print(f"Found total {len(all_images)} images and {len(all_annotations)} annotations")
    
    img_dir = os.path.join(input_dir, "train-0", "publaynet", "train")
    for img in tqdm(all_images, desc="Checking existing images"):
        src_path = os.path.join(img_dir, img['file_name'])
        if os.path.exists(src_path):
            existing_images.add(img['file_name'])
    
    filtered_images = [img for img in all_images if img['file_name'] in existing_images]
    filtered_annotations = [ann for ann in all_annotations 
                          if any(img['id'] == ann['image_id'] for img in filtered_images)]
    
    print(f"Found {len(existing_images)} existing images out of {len(all_images)}")
    
    for img in tqdm(filtered_images, desc="Copying images"):
        src_path = os.path.join(img_dir, img['file_name'])
        shutil.copy2(src_path, os.path.join(temp_img_dir, img['file_name']))
    
    combined_data = {
        "info": train_data['info'],
        "licenses": train_data['licenses'],
        "categories": train_data['categories'],
        "images": filtered_images,
        "annotations": filtered_annotations
    }
    
    with open(os.path.join(temp_ann_dir, "all.json"), 'w') as f:
        json.dump(combined_data, f)
    
    return combined_data

def create_coco_dataset(temp_dir, output_dir, train_ratio=0.8):
    with open(os.path.join(temp_dir, "annotations", "all.json"), 'r') as f:
        data = json.load(f)
    
    random.shuffle(data['images'])
    split_idx = int(len(data['images']) * train_ratio)
    train_images = data['images'][:split_idx]
    val_images = data['images'][split_idx:]
    
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    train_anns = [ann for ann in data['annotations'] if ann['image_id'] in train_img_ids]
    val_anns = [ann for ann in data['annotations'] if ann['image_id'] in val_img_ids]
    
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    train_data = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": train_images,
        "annotations": train_anns
    }
    
    val_data = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": val_images,
        "annotations": val_anns
    }
    
    with open(os.path.join(output_dir, "annotations", "instances_train.json"), 'w') as f:
        json.dump(train_data, f)
    
    with open(os.path.join(output_dir, "annotations", "instances_val.json"), 'w') as f:
        json.dump(val_data, f)
    
    for img in tqdm(train_images, desc="Copying train images"):
        src_path = os.path.join(temp_dir, "images", img['file_name'])
        dst_path = os.path.join(output_dir, "images", "train", img['file_name'])
        shutil.copy2(src_path, dst_path)
    
    for img in tqdm(val_images, desc="Copying val images"):
        src_path = os.path.join(temp_dir, "images", img['file_name'])
        dst_path = os.path.join(output_dir, "images", "val", img['file_name'])
        shutil.copy2(src_path, dst_path)
    
    print("\nDataset Summary:")
    print("---------------")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")
    print(f"Train set: {len(train_images)} images, {len(train_anns)} annotations")
    print(f"Val set: {len(val_images)} images, {len(val_anns)} annotations")
    print("\nDirectory structure:")
    print(f"{output_dir}/")
    print("├── images/")
    print("│   ├── train/")
    print("│   └── val/")
    print("└── annotations/")
    print("    ├── instances_train.json")
    print("    └── instances_val.json")

if __name__ == "__main__":
    base_dir = "."
    input_dir = os.path.join(base_dir, "smaller_publaynet_dataset")
    temp_dir = os.path.join(base_dir, "temp_dataset")
    output_dir = os.path.join(base_dir, "coco_dataset")
    
    for dir_path in [temp_dir, output_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    gather_all_data(input_dir, temp_dir)
    create_coco_dataset(temp_dir, output_dir)
    shutil.rmtree(temp_dir) 