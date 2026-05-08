import csv
import random
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision.io as io
from torchvision.utils import make_grid
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# =========================
# Configuration & Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "train_labels.csv"
TEST_CSV = DATA_DIR / "test_labels.csv"

CHECKPOINT_DIR = Path("outputs/checkpoints")
PREVIEW_DIR = Path("outputs/eraser_previews")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4

# =========================
# Dataset & DataLoaders
# =========================
class ClockEraserDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)
        

    def __getitem__(self, index):
            row = self.rows[index]
            
            input_tensor = io.read_image(row["analog_path"])
            target_tensor = io.read_image(row["analog_without_hands_path"])
    
            input_tensor = input_tensor.float() / 255.0
            target_tensor = target_tensor.float() / 255.0
    
            # אוגמנטציות מסונכרנות (צבע, בהירות) - מופעלות על שתי התמונות
            if random.random() > 0.5:
                brightness = random.uniform(0.6, 1.2)
                input_tensor = TF.adjust_brightness(input_tensor, brightness)
                target_tensor = TF.adjust_brightness(target_tensor, brightness)
    
            if random.random() > 0.5:
                contrast = random.uniform(0.7, 1.3)
                input_tensor = TF.adjust_contrast(input_tensor, contrast)
                target_tensor = TF.adjust_contrast(target_tensor, contrast)
    
            # ==========================================
            # הטריק למחיקת מחוגים אמיתיים (מופעל רק על הקלט!)
            # ==========================================
            if random.random() > 0.3:
                # הוספת טשטוש אקראי רק לתמונה עם המחוגים כדי לדמות צלליות ו-Anti-aliasing
                # kernel_size חובה להיות אי-זוגי (3 או 5)
                blur_amount = random.choice([3, 5])
                input_tensor = TF.gaussian_blur(input_tensor, kernel_size=blur_amount)
    
            return input_tensor, target_tensor

# =========================
# Model Architecture
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class MediumUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # הכפלנו את כמות הערוצים בכל שכבה
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck עמוק יותר
        self.bottleneck = DoubleConv(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.output_activation(self.output_layer(d1))

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.slice.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred_features = self.slice(pred)
        target_features = self.slice(target)
        return nn.functional.mse_loss(pred_features, target_features)

class CombinedReconstructionLoss(nn.Module):
    def __init__(self, mse_weight=0.1, perceptual_weight=0.2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.perceptual = VGGPerceptualLoss() 
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)
        mse_loss = self.mse(prediction, target)
        p_loss = self.perceptual(prediction, target)
        return l1_loss + (self.mse_weight * mse_loss) + (self.perceptual_weight * p_loss)

# =========================
# Training Utilities
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    
    # עטיפת ה-dataloader ב-tqdm
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}] Train", leave=False)
    
    for input_images, target_images in pbar:
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()
        predicted_images = model(input_images)
        loss = criterion(predicted_images, target_images)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input_images.size(0)
        
        # עדכון ה-Loss הנוכחי בתוך סרגל ההתקדמות
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}] Test ", leave=False)
    
    with torch.no_grad():
        for input_images, target_images in pbar:
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            
            predicted_images = model(input_images)
            loss = criterion(predicted_images, target_images)
            
            total_loss += loss.item() * input_images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return total_loss / len(dataloader.dataset)

def save_prediction_preview(model, dataloader, epoch, output_dir, device, num_images=4):
    model.eval()
    input_images, target_images = next(iter(dataloader))
    input_images = input_images.to(device)
    target_images = target_images.to(device)
    
    with torch.no_grad():
        predicted_images = model(input_images)
        
    input_images = input_images[:num_images].cpu()
    predicted_images = predicted_images[:num_images].cpu()
    target_images = target_images[:num_images].cpu()
    
    grid = make_grid(torch.cat([input_images, predicted_images, target_images], dim=0), nrow=num_images)
    preview_image = transforms.ToPILImage()(grid)
    output_path = output_dir / f"epoch_{epoch:03d}.png"
    preview_image.save(output_path)

# =========================
# Main Training Loop
# =========================
if __name__ == "__main__":
    train_dataset = ClockEraserDataset(TRAIN_CSV)
    test_dataset = ClockEraserDataset(TEST_CSV)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    model = MediumUNet().to(device)
    criterion = CombinedReconstructionLoss(mse_weight=0.0, perceptual_weight=1.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_test_loss = float("inf")
    history = []
    patience = 8
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        test_loss = evaluate(model, test_loader, criterion, device, epoch, NUM_EPOCHS)
        scheduler.step(test_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            checkpoint_path = CHECKPOINT_DIR / "best_eraser_unet.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model:", checkpoint_path)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

        if epoch == 1 or epoch % 5 == 0:
            save_prediction_preview(model, test_loader, epoch, PREVIEW_DIR, device)

    history_path = "outputs/eraser_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print("Saved history to:", history_path)