"""Module for processing image data using tf.data pipelines."""

import os
from typing import Tuple

import tensorflow as tf


def process_image(
    file_path: tf.Tensor, 
    img_size: Tuple[int, int] = (64, 64), 
    add_noise: bool = False
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads, resizes, and normalizes an image file. Optionally adds Gaussian noise.

    Args:
        file_path (tf.Tensor): Path to the image file.
        img_size (Tuple[int, int]): Target size for resizing. Defaults to (64, 64).
        add_noise (bool): Whether to add random noise to the input. Defaults to False.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple of (input_image, target_image).
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1) # Assuming grayscale medical images
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    
    if add_noise:
        # Add noise for denoising Autoencoder task
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.2)
        noisy_img = tf.clip_by_value(img + noise, 0.0, 1.0)
        return noisy_img, img
        
    return img, img


def get_dataset(
    base_data_path: str, 
    region_name: str, 
    batch_size: int = 32, 
    img_size: Tuple[int, int] = (64, 64),
    denoising: bool = False
) -> tf.data.Dataset:
    """
    Creates an optimized tf.data pipeline for a specific anatomical region.

    Args:
        base_data_path (str): The root path to the raw data directory.
        region_name (str): The name of the folder containing the region's images.
        batch_size (int): The batch size for training. Defaults to 32.
        img_size (Tuple[int, int]): The dimensions to resize images to.
        denoising (bool): Flag to apply noise for denoising tasks.

    Returns:
        tf.data.Dataset: A batched and prefetched TensorFlow dataset.
    """
    # Look for all files in the specific region folder
    region_path = os.path.join(base_data_path, region_name, '*.*')
    dataset = tf.data.Dataset.list_files(region_path, shuffle=True)
    
    dataset = dataset.map(
        lambda x: process_image(x, img_size, add_noise=denoising), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
