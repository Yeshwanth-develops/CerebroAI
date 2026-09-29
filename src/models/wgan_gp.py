"""
Improved Wasserstein GAN with Gradient Penalty (WGAN-GP) for Brain MRI Synthesis.
Implemented with modern TensorFlow 2.x Keras Model Subclassing.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int = 128, output_shape: Tuple[int, int, int] = (192, 160, 1)) -> keras.Model:
    """
    Generator network mapping latent vector z -> 192x160x1 synthetic MRI slice.
    """
    init_h, init_w = output_shape[0] // 8, output_shape[1] // 8

    inputs = layers.Input(shape=(latent_dim,), name="latent_noise")
    x = layers.Dense(init_h * init_w * 256, kernel_initializer="he_normal")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((init_h, init_w, 256))(x)

    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = layers.Conv2D(output_shape[2], (3, 3), padding="same", activation="tanh", name="generated_mri")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="WGAN_Generator")


def build_critic(input_shape: Tuple[int, int, int] = (192, 160, 1)) -> keras.Model:
    """
    Critic / Discriminator network scoring realism under 1-Lipschitz continuity.
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, name="critic_score")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="WGAN_Critic")


class WGAN_GP(keras.Model):
    """
    Full WGAN-GP training wrapper with gradient penalty enforcement.
    """

    def __init__(
        self,
        critic: keras.Model,
        generator: keras.Model,
        latent_dim: int = 128,
        critic_steps: int = 5,
        gp_weight: float = 10.0,
    ):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = critic_steps
        self.gp_weight = gp_weight

        self.c_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_tracker = keras.metrics.Mean(name="gp")

    @property
    def metrics(self):
        return [self.c_loss_tracker, self.g_loss_tracker, self.gp_tracker]

    def compile(self, c_optimizer, g_optimizer):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    def _gradient_penalty(self, batch_size: int, real_images: tf.Tensor, fake_images: tf.Tensor) -> tf.Tensor:
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)

                c_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self._gradient_penalty(batch_size, real_images, fake_images)
                c_loss = c_cost + self.gp_weight * gp

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.critic(generated_images, training=True)
            g_loss = -tf.reduce_mean(gen_img_logits)

        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        self.c_loss_tracker.update_state(c_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.gp_tracker.update_state(gp)
        return {
            "c_loss": self.c_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "gp": self.gp_tracker.result(),
        }
