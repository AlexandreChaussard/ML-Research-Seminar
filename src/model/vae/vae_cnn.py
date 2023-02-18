import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator
import src.utils.utils as utils
import ipdb


class VAE_Encoder(torch.nn.Module):
    def __init__(self, channels_size, z_dim, input_dim, strides):
        super(VAE_Encoder, self).__init__()

        init_channels = input_dim[0]
        channels_size = [init_channels] + channels_size

        self.conv_net = []
        for i in range(len(channels_size) - 1):
            try:
                self.conv_net.append(
                    nn.Conv2d(
                        channels_size[i], channels_size[i + 1], kernel_size=strides[i]
                    )
                )
                self.conv_net.append(
                    nn.BatchNorm2d(channels_size[i + 1])
                )
                # self.conv_net.append(nn.MaxPool2d(kernel_size=(2,2)))
                self.conv_net.append(nn.ReLU())
            except:
                ipdb.set_trace()

        self.conv_net = nn.Sequential(*self.conv_net)
        num_features_before_fcnn = functools.reduce(
            operator.mul, list(self.conv_net(torch.rand(1, *input_dim)).shape)
        )
        self.final_2d_shape = self.conv_net(torch.rand(1, *input_dim)).shape[1:]
        self.num_features_before_fcnn = num_features_before_fcnn
        self.linear_last = nn.Linear(num_features_before_fcnn, 200)
        self.mu_net = nn.Linear(200, z_dim)
        self.sigma_net = nn.Linear(200, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv_net(x)
        batch_size = x.shape[0]
        h = h.view([batch_size, -1])
        h = self.relu(self.linear_last(h))
        return self.mu_net(h), self.sigma_net(h)


class VAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        channels_size,
        z_dim,
        input_dim,
        strides,
        num_features_before_fcnn,
        final_2d_shape,
    ):
        super(VAE_Decoder, self).__init__()

        init_channels = final_2d_shape[0]
        channels_size = [init_channels] + channels_size
        self.channels_size = channels_size

        self.linear_decoder = nn.Sequential(
            nn.Linear(z_dim, 200), 
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, num_features_before_fcnn), nn.ReLU()
        )
        self.final_2d_shape = final_2d_shape
        self.conv_net = []
        for i in range(len(channels_size) - 1):
            print()
            self.conv_net.append(
                nn.ConvTranspose2d(
                    channels_size[i], channels_size[i + 1], kernel_size=strides[i]
                )
            )
            self.conv_net.append(
                nn.BatchNorm2d(channels_size[i + 1])
            )
            # self.conv_net.append(nn.MaxUnpool2d(kernel_size=(2,2)))
            if i < len(channels_size) - 2:
                self.conv_net.append(nn.ReLU())
            else:
                # self.conv_net.append(nn.Sigmoid())
                pass

        self.conv_net = nn.Sequential(*self.conv_net)

        # self.output_net = nn.Linear(hidden_sizes[-1], input_dim)

    def forward(self, z: torch.Tensor):
        h_bis = self.linear_decoder(z).view(
            [-1, self.final_2d_shape[0], self.final_2d_shape[1], self.final_2d_shape[2]]
        )
        # ipdb.set_trace()
        h = self.conv_net(h_bis)
        # return F.sigmoid().view(
        #     -1, self.n_channels, self.n_rows, self.n_cols
        # )
        return h

class VAE(torch.nn.Module):
    def __init__(
        self,
        conv_2d_channels_encoder,
        conv_2d_channels_decoder,
        z_dim,
        input_dim,
        strides
    ):
        super(VAE, self).__init__()

        self.z_dim = z_dim

        self.encoder = VAE_Encoder(
            conv_2d_channels_encoder, z_dim, input_dim, strides
        )
        self.num_before_fcnn = self.encoder.num_features_before_fcnn
        self.final_2d_shape = self.encoder.final_2d_shape

        self.decoder = VAE_Decoder(
            conv_2d_channels_decoder,
            z_dim, 
            input_dim,
            strides,
            self.num_before_fcnn,
            self.final_2d_shape
        )

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)
        z = utils.sampling(z_mu, z_log_var)
        return self.decoder(z), z_mu, z_log_var

    def loss_function(self, x, y, mu, log_var):
        batch_size = x.shape[0]
        reconstruction_error = F.binary_cross_entropy(
            y.view([batch_size, -1]), x.view([batch_size, -1]), reduction="sum"
        )

        KLD = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - 1 - log_var)

        return reconstruction_error + KLD


class VAE_CNN:

    def __init__(
        self,
        model
    ):
        self.model = model

    def loss_function(self, x, y, mu, log_var,device):
        batch_size = x.shape[0]
        reconstruction_error = F.binary_cross_entropy(
            y.view([batch_size, -1]), x.view([batch_size, -1]), reduction="sum"
        )
        # reconstruction_error = ((y-x)**2).mean() 
        reconstruction_error = reconstruction_error.to(device)
        
        KLD = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - 1 - log_var)
        KLD = KLD.to(device)
        return reconstruction_error + KLD

    def train_vae(self, learning_rate, data_train_loader,n_epochs):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            self.model = self.model.cuda()
            self.device = device

        for epoch in range(n_epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(data_train_loader):
                optimizer.zero_grad()

                data = data.to(device)
                y, z_mu, z_log_var = self.model(data)
                y = y.to(device)
                z_mu = z_mu.to(device)
                z_log_var = z_log_var.to(device)
                loss_vae = self.loss_function(data, y, z_mu, z_log_var,device)
                loss_vae.backward()
                train_loss += loss_vae.item()
                optimizer.step()

            print(
                "[*] Epoch: {} Average loss: {:.4f}".format(
                    epoch, train_loss / len(data_train_loader.dataset)
                )
            )
    
    def generate_data(self, n_data=5):
        epsilon = torch.randn(n_data, 1, self.model.z_dim).to(self.device)
        generations = self.model.decoder(epsilon)
        return generations


def train_vae(vae_model, optimizer, data_train_loader, n_epoch):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()

            y, z_mu, z_log_var = vae_model(data)
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print(
            "[*] Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(data_train_loader.dataset)
            )
        )

    return vae_model


def generate_data(vae_model, n_data=5):
    epsilon = torch.randn(n_data, 1, vae_model.z_dim)
    generations = vae_model.decoder(epsilon)
    return generations


def train_vae_inverse_noise(
    vae_model, optimizer, data_train_loader, n_epoch, noise_mean, noise_std
):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a gaussian noise to the data in entry, the goal being to reconstruct it
            noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(noisy_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print(
            "[*] Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(data_train_loader.dataset)
            )
        )

    return vae_model


def restore_noisy_data(vae_model, clean_data_loader, noise_mean, noise_std):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_vae_inverse_lostdata(
    vae_model, optimizer, data_train_loader, n_epoch, square_size
):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_add_square(data, square_size)
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(lost_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print(
            "[*] Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(data_train_loader.dataset)
            )
        )

    return vae_model


def restore_lostdata_data(vae_model, clean_data_loader, square_size):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_add_square(data, square_size)
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_vae_inverse_blur(
    vae_model, optimizer, data_train_loader, n_epoch, kernel_size, sigma
):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_gaussian_blur(
                data, kernel_size=kernel_size, sigma=sigma
            )
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(lost_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print(
            "[*] Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(data_train_loader.dataset)
            )
        )

    return vae_model


def restore_blur_data(vae_model, clean_data_loader, kernel_size, sigma):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_gaussian_blur(
            data, kernel_size=kernel_size, sigma=sigma
        )
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list
