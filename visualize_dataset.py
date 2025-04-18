import os
import json
import cv2
import numpy as np
from tqdm import tqdm

class COCOVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_dir = os.path.join(dataset_path, "annotations")
        
        with open(os.path.join(self.annotations_dir, "instances_train.json"), 'r') as f:
            self.train_data = json.load(f)
        
        with open(os.path.join(self.annotations_dir, "instances_val.json"), 'r') as f:
            self.val_data = json.load(f)
        
        self.categories = {cat['id']: cat['name'] for cat in self.train_data['categories']}
        self.colors = {
            cat_id: tuple(np.random.randint(0, 255, 3).tolist())
            for cat_id in self.categories.keys()
        }
        
        self.current_set = 'train'
        self.current_idx = 0
        self.images = self.train_data['images']
        self.annotations = self.train_data['annotations']
    
    def switch_dataset(self, set_name):
        if set_name == 'train':
            self.images = self.train_data['images']
            self.annotations = self.train_data['annotations']
        else:
            self.images = self.val_data['images']
            self.annotations = self.val_data['annotations']
        self.current_set = set_name
        self.current_idx = 0
    
    def get_current_image(self):
        if self.current_idx >= len(self.images):
            return None, None
        
        img_info = self.images[self.current_idx]
        img_path = os.path.join(self.images_dir, self.current_set, img_info['file_name'])
        img = cv2.imread(img_path)
        
        img_anns = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        
        return img, img_anns
    
    def draw_annotations(self, img, annotations):
        for ann in annotations:
            bbox = ann['bbox']
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]
            color = self.colors[cat_id]
            
            x, y, w, h = [int(coord) for coord in bbox]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            label = f"{cat_name}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - label_height - 10), (x + label_width, y), color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def show_next(self):
        self.current_idx = (self.current_idx + 1) % len(self.images)
        return self.show_current()
    
    def show_previous(self):
        self.current_idx = (self.current_idx - 1) % len(self.images)
        return self.show_current()
    
    def show_current(self):
        img, annotations = self.get_current_image()
        if img is None:
            return None
        
        img_with_anns = self.draw_annotations(img.copy(), annotations)
        
        img_info = self.images[self.current_idx]
        status_text = f"Set: {self.current_set} | Image: {self.current_idx + 1}/{len(self.images)} | ID: {img_info['id']}"
        cv2.putText(img_with_anns, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_with_anns

def main():
    dataset_path = "coco_dataset"
    visualizer = COCOVisualizer(dataset_path)
    
    print("\nControls:")
    print("n - Next image")
    print("p - Previous image")
    print("t - Switch to train set")
    print("v - Switch to validation set")
    print("q - Quit")
    
    window_name = "COCO Dataset Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        img = visualizer.show_current()
        if img is None:
            break
        
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):
            img = visualizer.show_next()
        elif key == ord('p'):
            img = visualizer.show_previous()
        elif key == ord('t'):
            visualizer.switch_dataset('train')
        elif key == ord('v'):
            visualizer.switch_dataset('val')
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 