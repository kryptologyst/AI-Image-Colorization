import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

# Import our modules
from modern_colorization import (
    ColorizationUNet, 
    ColorizationDataset, 
    ColorizationTrainer,
    create_mock_dataset,
    lab_to_rgb
)

class TestColorizationUNet:
    """Test the ColorizationUNet model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = ColorizationUNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = ColorizationUNet()
        input_tensor = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 2, 256, 256)
        assert torch.is_tensor(output)
    
    def test_model_parameters(self):
        """Test model has trainable parameters"""
        model = ColorizationUNet()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        assert num_params < 100_000_000  # Reasonable size limit

class TestColorizationDataset:
    """Test the ColorizationDataset class"""
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized"""
        # Create temporary image
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image_path = os.path.join(temp_dir, 'test_image.jpg')
            cv2.imwrite(image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
            
            dataset = ColorizationDataset([image_path])
            assert len(dataset) == 1
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image_path = os.path.join(temp_dir, 'test_image.jpg')
            cv2.imwrite(image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
            
            dataset = ColorizationDataset([image_path])
            sample = dataset[0]
            
            assert 'L' in sample
            assert 'AB' in sample
            assert 'original' in sample
            assert torch.is_tensor(sample['L'])
            assert torch.is_tensor(sample['AB'])
            assert torch.is_tensor(sample['original'])

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_lab_to_rgb_conversion(self):
        """Test LAB to RGB conversion"""
        # Create dummy LAB data
        L = np.random.rand(256, 256) * 100
        AB = np.random.rand(256, 256, 2) * 255 - 128
        
        rgb = lab_to_rgb(L, AB)
        
        assert rgb.shape == (256, 256, 3)
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 1)
    
    def test_create_mock_dataset(self):
        """Test mock dataset creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_mock_dataset(num_images=5, output_dir=temp_dir)
            
            # Check if images were created
            image_files = list(Path(temp_dir).glob('*.jpg'))
            assert len(image_files) == 5
            
            # Check if images are valid
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                assert img is not None
                assert img.shape[2] == 3  # 3 channels

class TestColorizationTrainer:
    """Test the ColorizationTrainer class"""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized"""
        model = ColorizationUNet()
        trainer = ColorizationTrainer(model)
        
        assert trainer.model is not None
        assert trainer.device is not None
        assert trainer.optimizer is not None
        assert trainer.criterion_mse is not None
    
    def test_trainer_device_selection(self):
        """Test trainer selects correct device"""
        model = ColorizationUNet()
        trainer = ColorizationTrainer(model)
        
        # Should select CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            assert trainer.device.type == 'cuda'
        else:
            assert trainer.device.type == 'cpu'

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock dataset
            create_mock_dataset(num_images=2, output_dir=temp_dir)
            
            # Create dataset
            image_paths = [str(p) for p in Path(temp_dir).glob('*.jpg')]
            dataset = ColorizationDataset(image_paths)
            
            # Test data loading
            sample = dataset[0]
            assert sample['L'].shape == (1, 256, 256)
            assert sample['AB'].shape == (2, 256, 256)
            
            # Test model
            model = ColorizationUNet()
            input_tensor = sample['L'].unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, 2, 256, 256)

# Performance tests
class TestPerformance:
    """Performance tests"""
    
    def test_model_inference_speed(self):
        """Test model inference speed"""
        model = ColorizationUNet()
        input_tensor = torch.randn(1, 1, 256, 256)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 1 second per inference)
        assert avg_time < 1.0
    
    def test_memory_usage(self):
        """Test memory usage is reasonable"""
        model = ColorizationUNet()
        
        # Check model size
        model_size = sum(p.numel() for p in model.parameters()) * 4  # 4 bytes per float32
        assert model_size < 100 * 1024 * 1024  # Less than 100MB

if __name__ == "__main__":
    pytest.main([__file__])
