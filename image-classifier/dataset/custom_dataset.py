#custom_dataset.py
import os
import shutil
import random
from PIL import Image
from torch.utils.data import Dataset

# --- THE UTILITY LOGIC (Your current split logic) ---
def split_raw_into_final(src_root, dest_root, train_ratio=0.8):
    """
    Takes images from 'raw_dataset/class/' and splits them into 
    'final_dataset/train/class/' and 'final_dataset/test/class/'.
    """
    for mode in ['train', 'test']:
        os.makedirs(os.path.join(dest_root, mode), exist_ok=True)
    
    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(dest_root, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_root, 'test', cls), exist_ok=True)
        
        imgs = os.listdir(os.path.join(src_root, cls))
        random.shuffle(imgs)
        
        split_idx = int(len(imgs) * train_ratio)
        # Logic to copy images into the train and test folders
        for i, img in enumerate(imgs):
            target = 'train' if i < split_idx else 'test'
            shutil.copy2(os.path.join(src_root, cls, img), 
                         os.path.join(dest_root, target, cls, img))

# --- THE MANDATORY CLASS (Step 2 of PDF) ---
class CIFAR10Dataset(Dataset):
    """
    Requirement: Implement custom dataset class inheriting from torch Dataset[cite: 26].
    Overrides: __len__ and __getitem__[cite: 31, 32].
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples) # [cite: 31]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image) # [cite: 41, 42]
        return image, label # [cite: 32]
