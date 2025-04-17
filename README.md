# Key Optimizations for Speed

- **Reduced Dataset Size**: Using only 15% of the full dataset while maintaining class distribution (stratified sampling)  
- **Smaller Model**: Replaced ResNet50 with the more efficient EfficientNet-B0  
- **Reduced Image Resolution**: Using 128×128 images instead of 224×224  
- **Increased Batch Size**: Enlarged from 8 to 64 for faster processing  
- **Mixed Precision Training**: Using PyTorch's automatic mixed precision (AMP) for faster computation  
- **Frozen Early Layers**: Only training the final layers of the network to reduce the number of parameters  
- **Optimized DataLoader**: Added persistent workers, pinned memory, and increased number of workers  
- **Reduced Validation Frequency**: Only validating every 2 epochs (customizable)  
- **Early Stopping**: Implementation with a patience of 3 epochs to avoid unnecessary training  
- **Focused Metadata Features**: Using only the most important metadata features when in multi-modal mode  

# Code Structure

The code is organized into several key components:

## Dataset Classes

- `ISIC_HDF5_Dataset`: For image-only approach  
- `ISIC_MultiModal_Dataset`: For combined image + metadata approach  

## Model Architecture

- `LightSkinLesionModel`: A lightweight model based on EfficientNet-B0 with optional metadata integration  

## Training Functions

- `train_epoch`: Handles one training epoch with mixed precision  
- `evaluate`: For validation with AUC calculation  
- `predict`: For generating predictions on test data  

## Main Workflow

- `setup_data_and_model`: Prepares datasets, dataloaders, and model  
- `main`: Orchestrates the entire training and prediction pipeline  
