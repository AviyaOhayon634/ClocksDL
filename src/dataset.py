import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class TimepieceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired digital and analog clock faces.
    Expects architecture: base_path -> [analog|digital] -> partition
    """
    def __init__(self, base_path: str, partition: str = "train", transforms_pipeline=None):
        super().__init__()
        self.base_path = base_path
        self.partition = partition
        self.transforms_pipeline = transforms_pipeline
        
        # Define exact paths to the newly separated CSV files
        self.dig_csv_path = os.path.join(base_path, "digital", partition, "labels.csv")
        self.ana_csv_path = os.path.join(base_path, "analog", partition, "labels.csv")
        
        # Validate existence
        if not os.path.exists(self.dig_csv_path):
            raise FileNotFoundError(f"Missing digital records: {self.dig_csv_path}")
        if not os.path.exists(self.ana_csv_path):
            raise FileNotFoundError(f"Missing analog records: {self.ana_csv_path}")
            
        # Load metadata
        self.digital_records = pd.read_csv(self.dig_csv_path)
        self.analog_records = pd.read_csv(self.ana_csv_path)
        
        # Safety check to ensure both folders generated the same amount of samples
        if len(self.digital_records) != len(self.analog_records):
            raise ValueError("Mismatch in dataset sizes between analog and digital records!")

    def __len__(self):
        return len(self.digital_records)

    def __getitem__(self, index: int):
        rec_dig = self.digital_records.iloc[index]
        rec_ana = self.analog_records.iloc[index]
        
        # 1. Extract filenames using the new CSV headers
        file_dig = str(rec_dig["filename"])
        file_ana = str(rec_ana["filename"])
        file_clean = str(rec_ana["clean_filename"]) if "clean_filename" in rec_ana else None
        
        # 2. Construct precise file paths
        path_d = os.path.join(self.base_path, "digital", self.partition, file_dig)
        path_a = os.path.join(self.base_path, "analog", self.partition, file_ana)
        
        # 3. Load image pixels into memory
        img_d = Image.open(path_d).convert("RGB")
        img_a = Image.open(path_a).convert("RGB")
        
        if file_clean:
            path_c = os.path.join(self.base_path, "analog", self.partition, file_clean)
            img_c = Image.open(path_c).convert("RGB")
        else:
            # Fallback if clean image isn't in the dataframe
            img_c = img_a.copy()
            
        # 4. Apply vision transforms
        if self.transforms_pipeline:
            img_d = self.transforms_pipeline(img_d)
            img_a = self.transforms_pipeline(img_a)
            img_c = self.transforms_pipeline(img_c)
            
        # 5. Extract time labels (using digital record, though both match)
        hr = int(rec_dig["hour"])
        mnt = int(rec_dig["minute"])
        sec = int(rec_dig["second"])
        
        # Format targets (0-1 regression bounds and raw integers)
        scaled_time = torch.tensor([hr / 23.0, mnt / 59.0, sec / 59.0], dtype=torch.float32)
        raw_time = torch.tensor([hr, mnt, sec], dtype=torch.long)
        
        # Return exact same dictionary structure
        return {
            "digital_img": img_d,
            "analog_img": img_a,
            "clean_img": img_c,  
            "time_label": scaled_time,
            "original_time": raw_time
        }