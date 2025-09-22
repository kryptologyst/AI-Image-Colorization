import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ColorizationDataset(Dataset):
    """Dataset for image colorization with modern data augmentation"""
    
    def __init__(self, image_paths, transform=None, mode='train'):
        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Extract L (grayscale) and AB (color) channels
        L = lab_image[:, :, 0]  # Grayscale channel
        AB = lab_image[:, :, 1:]  # Color channels
        
        # Normalize L channel to [0, 1]
        L = L.astype(np.float32) / 100.0
        
        # Normalize AB channels to [-1, 1]
        AB = AB.astype(np.float32)
        AB[:, :, 0] = (AB[:, :, 0] - 128) / 128.0  # A channel
        AB[:, :, 1] = (AB[:, :, 1] - 128) / 128.0  # B channel
        
        # Apply transforms if provided
        if self.transform:
            # Stack L and AB for albumentations
            stacked = np.concatenate([L[:, :, np.newaxis], AB], axis=2)
            transformed = self.transform(image=stacked)
            stacked = transformed['image']
            
            L = stacked[0]  # First channel
            AB = stacked[1:]  # Remaining channels
        
        return {
            'L': torch.tensor(L, dtype=torch.float32).unsqueeze(0),
            'AB': torch.tensor(AB, dtype=torch.float32),
            'original': torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        }

class ColorizationUNet(nn.Module):
    """Modern U-Net architecture for image colorization"""
    
    def __init__(self, input_channels=1, output_channels=2):
        super(ColorizationUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling path)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        output = self.final(dec1)
        return output

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG16
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.features = nn.ModuleList(vgg[:16])  # Use first 16 layers
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # Convert LAB to RGB for VGG
        pred_rgb = self.lab_to_rgb(pred)
        target_rgb = self.lab_to_rgb(target)
        
        # Extract features
        pred_features = pred_rgb
        target_features = target_rgb
        
        for layer in self.features:
            pred_features = layer(pred_features)
            target_features = layer(target_features)
        
        # Calculate MSE loss
        loss = F.mse_loss(pred_features, target_features)
        return loss
    
    def lab_to_rgb(self, lab_tensor):
        """Convert LAB tensor to RGB for VGG processing"""
        # This is a simplified conversion - in practice, you'd use proper LAB to RGB conversion
        return lab_tensor

class ColorizationTrainer:
    """Modern training pipeline with advanced techniques"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion_mse = nn.MSELoss()
        self.criterion_perceptual = PerceptualLoss().to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            L = batch['L'].to(self.device)
            AB = batch['AB'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_AB = self.model(L)
            
            # Calculate losses
            mse_loss = self.criterion_mse(pred_AB, AB)
            perceptual_loss = self.criterion_perceptual(pred_AB, AB)
            
            # Combined loss
            total_loss_batch = mse_loss + 0.1 * perceptual_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                L = batch['L'].to(self.device)
                AB = batch['AB'].to(self.device)
                
                pred_AB = self.model(L)
                loss = self.criterion_mse(pred_AB, AB)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

def create_mock_dataset(num_images=1000, output_dir='mock_dataset'):
    """Create a mock dataset for training"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} mock images...")
    
    for i in tqdm(range(num_images)):
        # Create random colorful image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Create random shapes
        center = (np.random.randint(50, 206), np.random.randint(50, 206))
        radius = np.random.randint(20, 80)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.circle(img, center, radius, color, -1)
        
        # Add random rectangles
        pt1 = (np.random.randint(0, 200), np.random.randint(0, 200))
        pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img, pt1, pt2, color, -1)
        
        # Save image
        cv2.imwrite(os.path.join(output_dir, f'image_{i:04d}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Mock dataset created in {output_dir}/")

def lab_to_rgb(L, AB):
    """Convert LAB to RGB"""
    # Ensure L has the right shape
    if L.ndim == 2:
        L = L[:, :, np.newaxis]
    
    # Combine L and AB channels
    lab = np.concatenate([L, AB], axis=2)
    
    # Convert LAB to RGB
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0, 1)

def visualize_results(model, dataloader, device, num_samples=4):
    """Visualize colorization results"""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            L = batch['L'].to(device)
            AB_true = batch['AB'].to(device)
            original = batch['original']
            
            # Get prediction
            AB_pred = model(L)
            
            # Convert to numpy
            L_np = L[0].cpu().numpy().squeeze()
            AB_true_np = AB_true[0].cpu().numpy().transpose(1, 2, 0)
            AB_pred_np = AB_pred[0].cpu().numpy().transpose(1, 2, 0)
            original_np = original[0].numpy().transpose(1, 2, 0)
            
            # Denormalize
            L_np = L_np * 100
            AB_true_np[:, :, 0] = AB_true_np[:, :, 0] * 128 + 128
            AB_true_np[:, :, 1] = AB_true_np[:, :, 1] * 128 + 128
            AB_pred_np[:, :, 0] = AB_pred_np[:, :, 0] * 128 + 128
            AB_pred_np[:, :, 1] = AB_pred_np[:, :, 1] * 128 + 128
            
            # Convert to RGB
            rgb_true = lab_to_rgb(L_np, AB_true_np)
            rgb_pred = lab_to_rgb(L_np, AB_pred_np)
            
            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(L_np, cmap='gray')
            axes[0].set_title('Grayscale (L)')
            axes[0].axis('off')
            
            axes[1].imshow(original_np)
            axes[1].set_title('Original')
            axes[1].axis('off')
            
            axes[2].imshow(rgb_true)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            axes[3].imshow(rgb_pred)
            axes[3].set_title('Predicted')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.show()

def main():
    """Main training function"""
    print("🚀 Starting Modern Image Colorization Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock dataset
    create_mock_dataset(num_images=500)
    
    # Get image paths
    image_paths = [f'mock_dataset/image_{i:04d}.jpg' for i in range(500)]
    
    # Define transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1]),
        ToTensorV2()
    ])
    
    # Split dataset
    train_paths = image_paths[:400]
    val_paths = image_paths[400:]
    
    # Create datasets
    train_dataset = ColorizationDataset(train_paths, transform=train_transform)
    val_dataset = ColorizationDataset(val_paths, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model and trainer
    model = ColorizationUNet()
    trainer = ColorizationTrainer(model, device)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Update learning rate
        trainer.scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_colorization_model.pth')
            print("💾 Best model saved!")
        
        # Visualize results every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device)
    
    print("🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
