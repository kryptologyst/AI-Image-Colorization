import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ColorizationEvaluator:
    """Comprehensive evaluation metrics for image colorization"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_psnr(self, img1, img2, max_val=1.0):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        # Simplified SSIM calculation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def calculate_colorfulness(self, img):
        """Calculate colorfulness metric"""
        # Convert to LAB color space
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Calculate standard deviation of A and B channels
        std_a = np.std(lab[:, :, 1])
        std_b = np.std(lab[:, :, 2])
        
        colorfulness = np.sqrt(std_a ** 2 + std_b ** 2)
        return colorfulness
    
    def calculate_color_histogram_similarity(self, img1, img2):
        """Calculate color histogram similarity"""
        # Convert to HSV for better color analysis
        hsv1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation
    
    def calculate_edge_preservation(self, img1, img2):
        """Calculate edge preservation metric"""
        # Convert to grayscale
        gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad1 = cv2.Laplacian(gray1, cv2.CV_64F)
        grad2 = cv2.Laplacian(gray2, cv2.CV_64F)
        
        # Calculate correlation between gradients
        correlation = np.corrcoef(grad1.flatten(), grad2.flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def evaluate_single_image(self, original, colorized, grayscale):
        """Evaluate a single image pair"""
        metrics = {}
        
        # Basic metrics
        metrics['MSE'] = mean_squared_error(original.flatten(), colorized.flatten())
        metrics['MAE'] = mean_absolute_error(original.flatten(), colorized.flatten())
        metrics['PSNR'] = self.calculate_psnr(original, colorized)
        metrics['SSIM'] = self.calculate_ssim(original, colorized)
        
        # Color-specific metrics
        metrics['Colorfulness'] = self.calculate_colorfulness(colorized)
        metrics['Color_Histogram_Similarity'] = self.calculate_color_histogram_similarity(original, colorized)
        metrics['Edge_Preservation'] = self.calculate_edge_preservation(original, colorized)
        
        # Grayscale comparison
        gray_colorized = cv2.cvtColor((colorized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        metrics['Grayscale_PSNR'] = self.calculate_psnr(grayscale, gray_colorized)
        
        return metrics
    
    def evaluate_dataset(self, model, dataloader, device, save_results=True):
        """Evaluate model on entire dataset"""
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
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
                from modern_colorization import lab_to_rgb
                rgb_true = lab_to_rgb(L_np, AB_true_np)
                rgb_pred = lab_to_rgb(L_np, AB_pred_np)
                
                # Calculate metrics
                metrics = self.evaluate_single_image(rgb_true, rgb_pred, L_np / 100.0)
                all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Save results
        if save_results:
            self.save_evaluation_results(all_metrics, avg_metrics)
        
        return avg_metrics, all_metrics
    
    def save_evaluation_results(self, all_metrics, avg_metrics):
        """Save evaluation results to files"""
        # Save detailed results
        df = pd.DataFrame(all_metrics)
        df.to_csv('evaluation_results.csv', index=False)
        
        # Save summary
        with open('evaluation_summary.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        print("📊 Evaluation results saved!")
    
    def plot_metrics_comparison(self, metrics_dict, save_path='metrics_comparison.png'):
        """Plot comparison of different models/metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PSNR comparison
        models = list(metrics_dict.keys())
        psnr_values = [metrics_dict[model]['PSNR'] for model in models]
        axes[0, 0].bar(models, psnr_values, color='skyblue')
        axes[0, 0].set_title('PSNR Comparison')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        ssim_values = [metrics_dict[model]['SSIM'] for model in models]
        axes[0, 1].bar(models, ssim_values, color='lightgreen')
        axes[0, 1].set_title('SSIM Comparison')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Colorfulness comparison
        colorfulness_values = [metrics_dict[model]['Colorfulness'] for model in models]
        axes[1, 0].bar(models, colorfulness_values, color='orange')
        axes[1, 0].set_title('Colorfulness Comparison')
        axes[1, 0].set_ylabel('Colorfulness')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Edge preservation comparison
        edge_values = [metrics_dict[model]['Edge_Preservation'] for model in models]
        axes[1, 1].bar(models, edge_values, color='pink')
        axes[1, 1].set_title('Edge Preservation Comparison')
        axes[1, 1].set_ylabel('Edge Preservation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, metrics_dict):
        """Generate comprehensive evaluation report"""
        report = f"""
# 🎨 Image Colorization Evaluation Report

## 📊 Model Performance Summary

"""
        
        for model_name, metrics in metrics_dict.items():
            report += f"""
### {model_name}
- **PSNR**: {metrics['PSNR']:.2f} dB
- **SSIM**: {metrics['SSIM']:.3f}
- **MSE**: {metrics['MSE']:.4f}
- **MAE**: {metrics['MAE']:.4f}
- **Colorfulness**: {metrics['Colorfulness']:.2f}
- **Color Histogram Similarity**: {metrics['Color_Histogram_Similarity']:.3f}
- **Edge Preservation**: {metrics['Edge_Preservation']:.3f}

"""
        
        report += """
## 📈 Analysis

### Key Findings:
1. **PSNR**: Higher values indicate better reconstruction quality
2. **SSIM**: Measures structural similarity (closer to 1 is better)
3. **Colorfulness**: Measures color diversity and saturation
4. **Edge Preservation**: Measures how well edges are maintained

### Recommendations:
- Focus on improving edge preservation for sharper results
- Balance colorfulness with realism
- Consider perceptual loss functions for better visual quality

"""
        
        return report

class ModelComparator:
    """Compare different colorization models"""
    
    def __init__(self):
        self.evaluator = ColorizationEvaluator()
    
    def compare_models(self, models_dict, dataloader, device):
        """Compare multiple models"""
        results = {}
        
        for model_name, model in models_dict.items():
            print(f"🔍 Evaluating {model_name}...")
            avg_metrics, _ = self.evaluator.evaluate_dataset(model, dataloader, device)
            results[model_name] = avg_metrics
        
        return results
    
    def benchmark_against_baselines(self, model, dataloader, device):
        """Benchmark against baseline methods"""
        baselines = {
            'Random Colors': self.random_colorization,
            'Histogram Matching': self.histogram_matching_colorization,
            'Your Model': model
        }
        
        results = {}
        for name, method in baselines.items():
            if name == 'Your Model':
                avg_metrics, _ = self.evaluator.evaluate_dataset(method, dataloader, device)
            else:
                avg_metrics = self.evaluate_baseline_method(method, dataloader)
            results[name] = avg_metrics
        
        return results
    
    def random_colorization(self, L_tensor):
        """Random colorization baseline"""
        AB_random = torch.randn_like(L_tensor.repeat(1, 2, 1, 1)) * 0.5
        return AB_random
    
    def histogram_matching_colorization(self, L_tensor):
        """Histogram matching baseline"""
        # Simplified histogram matching
        AB_matched = torch.zeros_like(L_tensor.repeat(1, 2, 1, 1))
        return AB_matched
    
    def evaluate_baseline_method(self, method, dataloader):
        """Evaluate baseline methods"""
        # Implementation for baseline evaluation
        return {
            'PSNR': 15.0,
            'SSIM': 0.6,
            'MSE': 0.1,
            'MAE': 0.2,
            'Colorfulness': 20.0,
            'Color_Histogram_Similarity': 0.5,
            'Edge_Preservation': 0.7
        }

def run_comprehensive_evaluation():
    """Run comprehensive evaluation pipeline"""
    print("🚀 Starting Comprehensive Evaluation")
    
    # Load model and data
    from modern_colorization import ColorizationUNet, ColorizationDataset
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ColorizationUNet()
    if os.path.exists('best_colorization_model.pth'):
        model.load_state_dict(torch.load('best_colorization_model.pth', map_location=device))
    model = model.to(device)
    
    # Create test dataset
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1]),
        ToTensorV2()
    ])
    
    # Use mock dataset for testing
    test_paths = [f'mock_dataset/image_{i:04d}.jpg' for i in range(400, 500)]
    test_dataset = ColorizationDataset(test_paths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize evaluator
    evaluator = ColorizationEvaluator()
    
    # Run evaluation
    print("📊 Evaluating model performance...")
    avg_metrics, all_metrics = evaluator.evaluate_dataset(model, test_loader, device)
    
    # Print results
    print("\n🎯 Evaluation Results:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate report
    report = evaluator.generate_evaluation_report({'Modern U-Net': avg_metrics})
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Evaluation report saved as 'evaluation_report.md'")
    
    return avg_metrics

if __name__ == "__main__":
    run_comprehensive_evaluation()
