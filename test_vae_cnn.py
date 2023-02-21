import torch
import torch.nn as nn
import numpy as np
from src.model.vae.vae_cnn import *
from src.utils.viz import display_images

from src.data.dataloader import fetch_mnist_loader, fetch_cifar_loader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# %autoreload 2
z_dim = 200
conv_2d_channels = [32,64,64,32]
lr = 0.0005
dropout = 0
n_epoch = 600
strides = [(4,4),(4,4),(4,4),(4,4)]
input_dim = (3,32,32)
inverse_conv2d_channels = conv_2d_channels[::-1][1:] + [input_dim[0]]

if __name__=="__main__":
    x = torch.randn((2,1,28,28))
    vae_encoder = VAE_Encoder(conv_2d_channels, z_dim, input_dim, strides)
    num_before_fcnn = vae_encoder.num_features_before_fcnn
    final_2d_shape = vae_encoder.final_2d_shape
    # ipdb.set_trace()
    vae_decoder = VAE_Decoder(
        inverse_conv2d_channels, 
        z_dim, 
        input_dim, 
        strides, 
        num_before_fcnn,
        final_2d_shape
    )
    latent_values = torch.randn((2,1,z_dim))
    reconstructed_images = vae_decoder(latent_values)
    vae_full = VAE(
        conv_2d_channels,
        inverse_conv2d_channels,
        z_dim,
        input_dim,
        strides,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(vae_full.parameters(), lr=lr)
    mnist_train_loader, mnist_test_loader, (n_channels, n_rows, n_cols) = fetch_cifar_loader(
    n_samples_train=512*3,
    n_samples_test=1000,
    batch_size=512,
    path_to_data="/src/data/"
    )

    # model = train_vae(vae_full, optimizer, mnist_train_loader, n_epoch=n_epoch)
    vae_final = VAE_CNN(vae_full)
    vae_final.train_vae(
        learning_rate=lr,
        data_train_loader=mnist_train_loader,
        n_epochs=n_epoch
    )

    # Generate new samples
    
    # Display the results
    try:
        generated_imgs = vae_final.generate_data(5).cpu()
        display_images(generated_imgs)
    except:
        ipdb.set_trace()
    ipdb.set_trace()