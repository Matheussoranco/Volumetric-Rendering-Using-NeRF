import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf

tf.random.set_seed(42)

import keras
from keras import layers
import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20

url = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)
data = keras.utils.get_file(origin=url)

data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

plt.imshow(images[np.random.randint(low=0, high=num_images)])
plt.show()

def encode_position():
    """
    Codifica a posição em seu recurso de Fourier correspondente.

    Argumentos:
        x: A coordenada de entrada.

    Retorna:
        Fourier apresenta tensores de posição.
    """
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0**i * x))
    return tf.concat(positions, axis=-1)

def get_rays(height, width, focal, pose):
    """
    Calcula o ponto de origem e o vetor de direção dos raios.

    Argumentos:
        altura: Altura da imagem.
        largura: Largura da imagem.
        focal: A distância focal entre as imagens e a câmera.
        pose: A matriz de pose da câmera.

    Retorna:
        Tupla de ponto de origem e vetor de direção para raios.
    """
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )
    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)
    
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3. -1]
    
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))
    
    return (ray_origins, ray_directions)

def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    """
    Renderiza os raios e os achata.

    Argumentos:
        ray_origins: os pontos de origem dos raios.
        ray_directions: Os vetores unitários de direção dos raios.
        perto: O limite próximo da cena volumétrica.
        longe: O limite distante da cena volumétrica.
        num_samples: Número de pontos de amostra em um raio.
        rand: Escolha para randomizar a estratégia de amostragem.

    Retorna:
       Tupla de raios achatados e pontos de amostra em cada raio.
       """
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)