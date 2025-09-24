# TrxCNN: Deep Learning for Full-Length Transcriptome Classification

TrxCNN is a deep learning model designed for classifying full-length transcriptome sequences using convolutional neural networks with residual connections. This implementation uses PyTorch to provide an efficient and scalable solution for genomic sequence analysis.

## 📄 Paper Reference

This model is introduced in the paper: [Using Deep Learning to Classify Full-Length Transcriptome Sequences](https://ieeexplore.ieee.org/document/10385824)

## 🚀 Features

- **CNN-based Architecture**: Uses 1D convolutional layers with residual blocks for sequence classification
- **Dynamic Sequence Length Handling**: Supports variable-length sequences with efficient padding and masking
- **Bucket Batching**: Implements intelligent batching strategy to optimize training efficiency
- **GPU Acceleration**: Full CUDA support with mixed precision training
- **Customizable Hyperparameters**: Comprehensive hyperparameter configuration system

## 🛠 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Required Dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas
pip install torchtext
```

## 📁 Project Structure

```
TrxCNN/
├── README.md                 # Project documentation
├── train.py                  # Main training script
├── protcnn_model.py         # Model architecture definition
├── fastq_dataset.py         # Dataset class for FASTQ data processing
├── prepocess_raw_data.py    # Data preprocessing utilities  
├── hparams.py               # Hyperparameter configurations
├── utils.py                 # Utility functions and custom collate functions
└── train.sh                 # Training script for batch execution
```

## 🔧 Model Architecture

The TrxCNN model consists of:

1. **Input Layer**: Processes DNA sequences (A, T, C, G) with one-hot encoding
2. **Initial Convolution**: 1D convolution with masking for variable-length sequences
3. **Residual Blocks**: Multiple residual blocks with dilated convolutions
4. **Global Pooling**: Max pooling across the sequence dimension
5. **Classification Layer**: Fully connected layer for transcript classification

### Key Components:

- **Conv1d_with_mask**: Custom convolution layer that handles variable-length sequences
- **Residual_Block**: Residual connection blocks with batch normalization and ReLU activation
- **DNA_Model**: Main model class combining all components

## 📊 Data Format

The model expects DNA sequence data in the following format:

- **Input**: FASTQ format files containing DNA sequences
- **Processing**: Sequences are converted to integer encoding (A=0, T=1, C=2, G=3)
- **Output**: Classification labels for different transcripts

### Data Statistics:
- Genes: 19,813
- Transcripts: 57,899 (classification classes)
- Dataset size: ~16GB in CSV format

## 🚀 Usage

### Training

1. **Prepare your data**: Place FASTQ files in the appropriate directory structure
2. **Preprocess data**:
   ```bash
   python prepocess_raw_data.py
   ```
3. **Start training**:
   ```bash
   python train.py
   # or use the batch script
   bash train.sh
   ```

### Configuration

Modify hyperparameters in `hparams.py`:

```python
def hparams_set_train():
    hparams = {}
    hparams["filters"] = 800          # Number of filters
    hparams["kernel_size"] = 3        # Convolution kernel size
    hparams["num_layers"] = 4         # Number of residual blocks
    hparams["lr_rate"] = 0.0005       # Learning rate
    hparams["num_epochs"] = 40        # Training epochs
    hparams["bt_size"] = 32           # Batch size
    return hparams
```

### Model Testing

```python
# Test a saved model
from train import test_saved_model
results = test_saved_model()
```

## 🎯 Training Features

- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Dynamic learning rate with exponential decay
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Model Checkpointing**: Automatic model saving during training
- **Bucket Batching**: Efficient batching strategy for variable-length sequences

## 📈 Performance

The model achieves competitive performance on full-length transcriptome classification tasks. Training logs and model checkpoints are automatically saved in the `saved_model/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{trxcnn2024,
  title={Using Deep Learning to Classify Full-Length Transcriptome Sequences},
  author={[Authors]},
  journal={IEEE},
  year={2024},
  url={https://ieeexplore.ieee.org/document/10385824}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Paper](https://ieeexplore.ieee.org/document/10385824)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: Make sure to have sufficient GPU memory for training, as the model processes large genomic sequences. The training script automatically detects and uses CUDA when available. 
