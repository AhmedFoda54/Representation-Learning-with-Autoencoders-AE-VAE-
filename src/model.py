"""Module containing the architectures for Autoencoders and Variational Autoencoders."""

from typing import Tuple, Any

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_ae(
    input_shape: Tuple[int, int, int] = (64, 64, 1), 
    latent_dim: int = 32
) -> Tuple[Model, Model, Model]:
    """Builds a standard convolutional Autoencoder."""
    
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent_repr = layers.Dense(latent_dim, name="latent_space")(x)
    
    encoder = Model(inputs, latent_repr, name="ae_encoder")
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    # Assuming 64x64 input, after two stride-2 convs, spatial dim is 16x16
    x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    
    decoder = Model(latent_inputs, outputs, name="ae_decoder")
    
    # Autoencoder combined
    ae_outputs = decoder(encoder(inputs))
    autoencoder = Model(inputs, ae_outputs, name="autoencoder")
    
    return autoencoder, encoder, decoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the image."""
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """Variational Autoencoder class with custom training step for KL loss."""
    def __init__(self, encoder: Model, decoder: Model, **kwargs: Any):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # ---> ADD THIS NEW METHOD <---
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Standard forward pass: encode input to latent space, then decode."""
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)
    # -----------------------------

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> dict:
        x_input, x_target = data 
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_input)
            reconstruction = self.decoder(z)
            
            # Reconstruction Loss (Mean Squared Error)
            reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(x_target - reconstruction), axis=(1, 2, 3))
                )
            # KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_vae(
    input_shape: Tuple[int, int, int] = (64, 64, 1), 
    latent_dim: int = 32
) -> Tuple[Model, Model, Model]:
    """Builds a Variational Autoencoder."""
    
    # VAE Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(inputs, [z_mean, z_log_var, z], name="vae_encoder")
    
    # VAE Decoder (Architecture identical to AE decoder)
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    
    decoder = Model(latent_inputs, outputs, name="vae_decoder")
    
    vae = VAE(encoder, decoder)
    return vae, encoder, decoder
