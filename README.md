# Medical Image Compression and Generation: AE vs. VAE

## Overview
This repository contains the implementation of Standard Autoencoders (AE) and Variational Autoencoders (VAE) designed to compress, denoise, and generate medical images across multiple anatomical regions (e.g., Chest CT, Brain MRI). The project adheres strictly to standard Machine Learning Engineering conventions, utilizing object-oriented module design and optimized `tf.data` pipelines.

## Project Structure
```text
├── models/                   # Saved model weights (.h5) and metadata (.json)
├── notebooks/
│   └── experiment_notebook.ipynb # Visualizations, PCA plotting, and image generation
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # tf.data pipeline creation and preprocessing
│   ├── model.py              # AE and VAE Keras subclass architectures
│   └── train.py              # Main training loop and configuration
├── README.md
└── requirements.txt
```

## Requirements & Setup

### Clone the repository:
```bash
git clone https://github.com/AhmedFoda54/Representation-Learning-with-Autoencoders-AE-VAE.git
cd Representation-Learning-with-Autoencoders-AE-VAE
```

### Set up a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** Core dependencies include `tensorflow`, `matplotlib`, `scikit-learn`, and `pandas`.

## Usage

### Training the Models
Ensure your data is placed inside `data/raw/` (or update the `DATASET_PATH` in `train.py` if running on Kaggle/Colab). Execute the training script from the root directory:

```bash
python src/train.py
```

This will automatically train both AE and VAE models for each specified anatomical region and save the weights/metadata to the `models/` directory.

### Visualizations and Experiments
To view reconstructions, denoising performance, and latent space visualizations, open the experiment notebook:

```bash
jupyter notebook notebooks/experiment_notebook.ipynb
```
