import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ClockDataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None):
        self.root_dir = os.path.join(root_dir, subset)
        self.labels_path = os.path.join(self.root_dir, "labels.csv")
        
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"labels.csv not found at: {self.labels_path}")

        self.df = pd.read_csv(self.labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Paths
        d_fn = str(row["digital_filename"])
        a_fn = str(row["analog_filename"])
        # Clean filename is optional, if not present we can use the analog image as a placeholder for the clean image
        c_fn = str(row["analog_clean_filename"]) if "analog_clean_filename" in row else None

        dig_path = os.path.join(self.root_dir, "digital", d_fn)
        ana_path = os.path.join(self.root_dir, "analog", a_fn)
        
        # 2. Open Images
        digital_img = Image.open(dig_path).convert("RGB") 
        analog_img = Image.open(ana_path).convert("RGB")
        
        if c_fn:
            clean_path = os.path.join(self.root_dir, "analog", c_fn)
            clean_img = Image.open(clean_path).convert("RGB")
        else:
            # Fallback if clean image is not available, we can use the analog image as a placeholder
            clean_img = analog_img.copy()

        # 3. Apply Transforms
        if self.transform:
            digital_img = self.transform(digital_img)
            analog_img = self.transform(analog_img)
            clean_img = self.transform(clean_img)

        # 4. Parse Time Labels
        h = int(row["hour"])
        m = int(row["minute"])
        s = int(row["second"])
        
        # Normalized labels for regression (0-1 range)
        time_label = torch.tensor([h/23.0, m/59.0, s/59.0], dtype=torch.float32)

        return {
            "digital_img": digital_img,
            "analog_img": analog_img,
            "clean_img": clean_img,  
            "time_label": time_label,
            "original_time": torch.tensor([h, m, s], dtype=torch.long)
        }