# AI Image Colorization - Quick Start Guide

## 5-Minute Setup

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/ai-image-colorization.git
cd ai-image-colorization

# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/your-username/ai-image-colorization.git
cd ai-image-colorization

# Create environment
conda env create -f environment.yml
conda activate ai-colorization

# Launch the web app
streamlit run app.py
```

### Option 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/your-username/ai-image-colorization.git
cd ai-image-colorization

# Build and run
docker-compose up -d

# Access the app at http://localhost:8501
```

## First Steps

1. **Open the Web Interface**
   - Navigate to `http://localhost:8501`
   - You'll see the beautiful AI Colorization interface

2. **Upload an Image**
   - Click "Choose an image file"
   - Select a grayscale or color image
   - Click "🎨 Colorize Image"

3. **View Results**
   - See the original and colorized images side by side
   - Check quality metrics (PSNR, SSIM, etc.)
   - Download the colorized result

## Training Your Own Model

```bash
# Generate mock dataset
python modern_colorization.py

# Train the model (this will take some time)
python modern_colorization.py

# Evaluate the trained model
python evaluation.py
```

## 🔧 Configuration

### Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set model path
export MODEL_PATH=./models/best_colorization_model.pth

# Set data path
export DATA_PATH=./mock_dataset
```

### Model Parameters

Edit `modern_colorization.py` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Model architecture

## Understanding Results

### Quality Metrics

- **PSNR**: Higher is better (typically 20-40 dB)
- **SSIM**: Closer to 1.0 is better
- **Colorfulness**: Measures color diversity
- **Edge Preservation**: How well edges are maintained

### Troubleshooting

**Common Issues:**

1. **CUDA out of memory**
   - Reduce batch size in training
   - Use CPU mode: `device = torch.device('cpu')`

2. **Model not loading**
   - Check if `best_colorization_model.pth` exists
   - Train a new model first

3. **Poor colorization quality**
   - Train for more epochs
   - Adjust learning rate
   - Use better dataset

## Advanced Usage

### Custom Dataset

```python
# Create your own dataset
from modern_colorization import ColorizationDataset

# Your image paths
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']

# Create dataset
dataset = ColorizationDataset(image_paths, transform=your_transform)

# Use in training
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### API Usage

```python
from modern_colorization import ColorizationUNet, lab_to_rgb
import torch

# Load model
model = ColorizationUNet()
model.load_state_dict(torch.load('best_colorization_model.pth'))

# Colorize image
def colorize_image(image_path):
    # Your colorization code here
    pass
```


## What's Next?

- Try different images and see the results
- Experiment with the model parameters
- Contribute to the project
- Share your colorized images!


