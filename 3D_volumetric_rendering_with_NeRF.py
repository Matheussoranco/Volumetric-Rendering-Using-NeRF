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

def encode_position(x):
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
    height_width_focal = pose[:3, -1]
    
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

def map_fn(pose):
    """Mapeia a pose individual para raios achatados e pontos de amostra.

    Argumentos:
        pose: A matriz de pose da câmera.

    Retorna:
        Tupla de raios achatados e pontos de amostra correspondentes ao
        pose de câmera.
    """
    (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=NUM_SAMPLES,
        rand=True,
    )
    return (rays_flat, t_vals)

split_index = int(num_images * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

train_poses = poses[:split_index]
val_poses = poses[split_index:]

train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
train_ds = (
    training_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=AUTO)
validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
val_ds = (
    validation_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

def get_nerf_model(num_layers, num_pos):
    """
    Gera a rede neural NeRF.

    Argumentos:
        num_layers: O número de camadas MLP.
        num_pos: o número de dimensões da codificação posicional.

    Retorna:
        O modelo `keras`.
    """
    inputs = keras.Input(shape=(num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            x = layers.concatenate([x, inputs], axis=-1)
    outputs = layers.Dense(units=4)(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    """
    Gera a imagem RGB e o mapa de profundidade a partir da previsão do modelo.

    Argumentos:
        modelo: O modelo MLP que é treinado para prever o rgb e
            densidade de volume da cena volumétrica.
        rays_flat: Os raios achatados que servem como entrada para
            o modelo NeRF.
        t_vals: Os pontos de amostra para os raios.
        rand: Escolha para randomizar a estratégia de amostragem.
        treinar: se o modelo está em fase de treinamento ou teste.

    Retorna:
        Tupla de imagem RGB e mapa de profundidade.
    """
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)
    predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))
    
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])
    
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights * t_vals, axis=-2)
    
    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return(rgb, depth_map)        

class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model
        
    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")
        
    def train_step(self, inputs):
        (images, rays) = inputs
        (rays_flat, t_vals) = rays
        
        with tf.GradientTape() as tape:
            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images, rgb)
            
        trainable_variables = self.nerf_model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}
    
    def test_step(self, inputs):
            (images, rays) = inputs
            (rays_flat, t_vals) = rays

            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images, rgb)

            psnr = tf.image.psnr(images, rgb, max_val=1.0)

            self.loss_tracker.update_state(loss)
            self.psnr_metric.update_state(psnr)
            return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}
    @property
    def metrics(self):
        return[self.loss_tracker, self.psnr_metric]
    
test_imgs, test_rays = next(iter(train_ds))
test_rays_flat, test_t_vals = test_rays

loss_list = []

class TrainMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=test_rays_flat,
            t_vals=test_t_vals,
            rand=True,
            train=False,
        )

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")
        plt.show()
        plt.close()

num_pos = H * W * NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model)
model.compile(
    optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
)

if not os.path.exists("images"):
    os.makedirs("images")

model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[TrainMonitor()],
)


def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


create_gif("images/*.png", "training.gif")
