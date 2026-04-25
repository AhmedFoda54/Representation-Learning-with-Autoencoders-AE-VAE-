"""Main script to train AE and VAE models across different anatomical regions."""

import json
import os
import sys

# Ensure src module can be imported
sys.path.append('/kaggle/working')

from src.data_processing import get_dataset
from src.model import build_ae, build_vae


def train_models() -> None:
    """Trains models for all regions and saves weights and metadata."""
    
    # --- CONFIGURE THESE FOR YOUR DATASET ---
    # Example: "/kaggle/input/medical-mnist"
    DATASET_PATH = "/kaggle/input/datasets/andrewmvd/medical-mnist" 
    # Example: ['ChestCT', 'BrainMRI', 'KneeXray']
    REGIONS = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
    # ----------------------------------------

    models_dir = "/kaggle/working/models"
    
    config = {
        "batch_size": 32,
        "epochs": 15,
        "latent_dim": 32,
        "img_size": [64, 64],
        "version": "v1"
    }
    
    for region in REGIONS:
        print(f"\n{'='*40}")
        print(f"Training Models for Region: {region}")
        print(f"{'='*40}")
        
        # 1. Load Data
        dataset = get_dataset(
            base_data_path=DATASET_PATH, 
            region_name=region, 
            batch_size=config["batch_size"],
            img_size=tuple(config["img_size"])
        )
        
        # 2. Train Standard AE
        print(f"\n--- Training Standard AE ({region}) ---")
        ae_model, _, _ = build_ae(latent_dim=config["latent_dim"])
        ae_model.compile(optimizer='adam', loss='mse')
        ae_model.fit(dataset, epochs=config["epochs"])
        
        ae_path = os.path.join(models_dir, f'ae_{region}_{config["version"]}.weights.h5')
        ae_model.save_weights(ae_path)
        
        # 3. Train VAE
        print(f"\n--- Training VAE ({region}) ---")
        vae_model, _, _ = build_vae(latent_dim=config["latent_dim"])
        vae_model.compile(optimizer='adam')
        vae_model.fit(dataset, epochs=config["epochs"])
        # ---> ADD THIS LINE to initialize the network architecture before training <---
        vae_model.build(input_shape=(None, config["img_size"][0], config["img_size"][1], 1))
        
        vae_model.fit(dataset, epochs=config["epochs"])
        
        vae_path = os.path.join(models_dir, f'vae_{region}_{config["version"]}.weights.h5')
        vae_model.save_weights(vae_path)
        
        # 4. Save Metadata
        meta_path = os.path.join(models_dir, f'config_{region}_{config["version"]}.json')
        with open(meta_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    print("\nAll training complete. Models saved to /kaggle/working/models/")

if __name__ == "__main__":
    train_models()
