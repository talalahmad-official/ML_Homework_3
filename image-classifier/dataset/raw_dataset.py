#raw_dataset.py

import os
import shutil
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

def create_mapped_folder(csv_path, src_img_dir, dest_root_dir):
    """
    Creates a folder structure where images are moved into subfolders 
    named after their labels (e.g., raw_dataset/frog/1.png).
    """
    # 1. Load labels
    df = pd.read_csv(csv_path)
    
    # 2. Get unique classes and create subdirectories
    classes = df['label'].unique()
    for cls in classes:
        os.makedirs(os.path.join(dest_root_dir, cls), exist_ok=True)
    
    print(f"Organizing 50,000 images into {len(classes)} class folders...")

    # 3. Copy images into their respective class folders
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['id']
        label = row['label']
        
        src_path = os.path.join(src_img_dir, f"{img_id}.png")
        dst_path = os.path.join(dest_root_dir, label, f"{img_id}.png")
        
        # Using copy2 to preserve metadata, or move if you want to save space
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    print(f"Mapping complete! Data is now in: {dest_root_dir}")

class raw_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class subfolders.
            transform (callable, optional): Optional transform to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Automatically detect classes based on folder names
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build a list of (image_path, label_int) tuples
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_name in os.listdir(class_dir):
                path = os.path.join(class_dir, img_name)
                self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# To run the mapping logic directly:
if __name__ == "__main__":
    CSV_PATH = '/content/cifar_dataset/trainLabels.csv'
    SRC_DIR = '/content/cifar_dataset/train'
    DEST_DIR = '/content/raw_dataset'
    
    create_mapped_folder(CSV_PATH, SRC_DIR, DEST_DIR)
