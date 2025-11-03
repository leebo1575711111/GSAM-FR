# GSAM-reg-main Time Series Forecasting Project

## Project Overview

This project implements time series forecasting models with advanced optimization strategies, including Sharpness-Aware Minimization (SAM) and Gradient-based Sharpness-Aware Minimization with Fisher-Rao Regularization (GSAM-FR). The project provides efficient solutions for multivariate long-term time series forecasting with support for multiple model architectures.

## Key Features

- 🚀 **Multiple Model Architectures** - Support for TSMixer, Transformer, and custom base models
- 💡 **Advanced Optimization Strategies** - SAM and GSAM-FR optimization for improved generalization
- 🔧 **Reversible Instance Normalization** - Handles distribution shifts between training and testing data
- ⚡ **Flexible Configuration** - Configurable optimization strategies (0=Adam, 1=SAM, 2=GSAM-FR)
- 📊 **Multiple Dataset Support** - Supports various time series datasets including ETT, weather, electricity, traffic, and more

## Project Structure

```
GSAM-reg-main--/
├── models/           # Model definitions
│   ├── base_model.py     # Base model class
│   ├── tsmixer_rev_in.py # TSMixer model implementation
│   └── utils/           # Model utility modules
├── utils/            # Utility modules
│   ├── callbacks.py      # Training callback functions
│   ├── data_utils.py     # Data processing utilities
│   ├── model_utils.py    # Model utilities
│   ├── train.py          # Training script
│   └── MemoryMonitor.py  # Memory monitoring tool
├── dataset/          # Dataset directory
│   └── ETTh1.csv         # Example dataset
├── run.py            # Main execution script
├── run_script.sh     # Batch experiment script
├── requirements.txt  # Dependency list
└── LICENSE           # License file
```

## Quick Start

### Environment Setup

```bash
# Clone the project
git clone <project-url>
cd GSAM-reg-main--

# Install dependencies
pip install -r requirements.txt
```

### Run Single Experiment

```bash
# Run with default parameters
python run.py --model tsmixer --data ETTh1 --seq_len 96 --pred_len 96

# Enable SAM optimization
python run.py --model tsmixer --data ETTh1 --opt_strategy 1 --rho 0.7

# Enable GSAM-FR optimization
python run.py --model tsmixer --data ETTh1 --opt_strategy 2 --rho 0.7 --alpha 0.5 --lambda_reg 0.1
```

### Batch Experiments

Use the provided script to run multiple prediction lengths and configurations:

```bash
# Run complete experiment pipeline
sh run_script.sh -m tsmixer -d ETTh1 -a

# Parameter explanation
-m Model name (tsmixer, etc.)
-d Dataset name (ETTh1, traffic, weather, etc.)
-s Sequence length (default 96)
-a Save additional results
```

## Supported Models and Datasets

### Models
- TSMixer (default)
- BaseModel (custom implementations)

### Datasets
- **ETT Series**: ETTh1, ETTh2, ETTm1, ETTm2
- **Electricity Data**: electricity
- **Weather Data**: weather  
- **Traffic Data**: traffic
- **Exchange Rate Data**: exchange_rate

## Configuration Parameters

Based on different datasets, the model automatically adjusts the following parameters:

| Dataset | Learning Rate | Blocks | Dropout | FF Dimension |
|---------|--------------|--------|---------|--------------|
| ETT Series | 0.001 | 2 | 0.9 | 64 |
| weather | 0.0001 | 4 | 0.3 | 32 |
| electricity | 0.0001 | 4 | 0.7 | 64 |
| traffic | 0.0001 | 8 | 0.7 | 64 |
| exchange_rate | 0.001 | 8 | 0.7 | 64 |

## Experimental Results

This project implements advanced optimization strategies (SAM and GSAM-FR) for time series forecasting models. The GSAM-FR optimization strategy provides improved generalization performance by combining gradient-based sharpness-aware minimization with Fisher-Rao regularization.

Key optimization strategies supported:
- **opt_strategy=0**: Standard Adam optimizer
- **opt_strategy=1**: SAM (Sharpness-Aware Minimization) optimization
- **opt_strategy=2**: GSAM-FR (Gradient-based SAM with Fisher-Rao Regularization)

## Custom Development

### Adding New Models

Create new model files in the `models/` directory and ensure proper import in `models/__init__.py`.

### Adding New Datasets

Place dataset files in the `dataset/` directory and add corresponding data processing logic in `utils/data_utils.py`.

### Custom Training Parameters

Modify parameter configurations in `run.py` or `run_script.sh`, or adjust directly through command-line arguments.

## License

This project is released under the MIT License.

Copyright 2024 Romain Ilbert  
Copyright 2025 Baofeng Liao

## Acknowledgments

This project builds upon the following excellent works:
- [SAMformer Paper](https://arxiv.org/pdf/2402.10198)
- [SAM Optimization Algorithm](https://github.com/google-research/sam)
- [TSMixer Model](https://github.com/google-research/google-research/tree/master/tsmixer)
- [RevIN Normalization](https://github.com/ts-kim/RevIN)

## Contact

For questions or suggestions, please contact:
- Project Maintainer: Baofeng Liao

---

**Start using GSAM-reg-main to experience efficient time series forecasting!**
