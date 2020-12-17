import torch
from torch.nn import functional as F
from torch import nn

# Our modules
from vae import configs


class Encoder(nn.Module):
    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Encoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_channels,
                               kernel_size=(7, 5),
                               stride=(3, 1),
                               padding=(1, 2))

        self.conv2 = nn.Conv2d(in_channels=self.num_channels,
                               out_channels=self.num_channels*2,
                               kernel_size=(5, 3),
                               stride=(3, 1),
                               padding=(1, 1))

        self.pool1 = nn.AdaptiveMaxPool2d((20, 1))

        self.fc_mu = nn.Linear(in_features=self.num_channels*2*20,
                               out_features=self.latent_dims)

        self.fc_logvar = nn.Linear(in_features=self.num_channels*2*20,
                                   out_features=self.latent_dims)

    def forward(self, x):
        global input_shape
        input_shape = x.shape[3]
        #print('Input size to Conv1 encoder:', x.shape)
        x_conv1 = F.relu(self.conv1(x))
        #print('Output size Conv1 encoder', x_conv1.shape)
        x_conv2 = F.relu(self.conv2(x_conv1))
        #print('Output size Conv2 encoder', x_conv2.shape)
        x_pool = self.pool1(x_conv2)
        #print('Output size after max-pool2', x_pool.shape)
        x_flatten = x_pool.view(x_pool.size(0), -1)  # flatten batch of feature maps
        #print('Output size after flatten', x_flatten.shape)

        x_mu = self.fc_mu(x_flatten)
        #print('Output size of mu after fc', x_mu.shape)
        x_logvar = self.fc_logvar(x_flatten)
        #print('Output size of logvar after fc', x_logvar.shape)
        return x_mu, x_logvar


class Decoder(nn.Module):
    global input_shape

    def __init__(self,
                 num_channels=configs.ParamsConfig.NUM_CHANNELS,
                 latent_dims=configs.ParamsConfig.LATENT_DIMS):
        super(Decoder, self).__init__()

        self.num_channels = num_channels
        self.latent_dims = latent_dims

        self.fc = nn.Linear(in_features=self.latent_dims,
                            out_features=self.num_channels*2*20)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels*2,
                                        out_channels=self.num_channels,
                                        kernel_size=(5, 3),
                                        stride=(3, 1),
                                        padding=(2, 1),
                                        dilation=(2, 1))

        self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels,
                                        out_channels=1,
                                        kernel_size=(7, 5),
                                        stride=(3, 1),
                                        padding=(3, 2),
                                        dilation=(2, 1))

    def forward(self, x):
        global input_shape

        #print('Input size to decoder:', x.shape)
        x = self.fc(x)
        #print('Output size of fc decoder:', x.shape)
        x = x.view(x.size(0), self.num_channels*2, 20, -1)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        #print('Output size after unflatten:', x.shape)
        x = nn.AdaptiveMaxPool2d((20, input_shape))(x)
        #print('Output size after unpool2:', x.shape)
        x = F.relu(self.conv2(x))
        #print('Output size after conv2:', x.shape)
        x = torch.sigmoid(self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #print('Output size after conv1 with Sigmoid:', x.shape)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
