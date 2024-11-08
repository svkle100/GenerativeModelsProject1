import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder
        self.encoder = torch.nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ])
        self.mu = nn.Linear(32 * (self.input_size//2**4)**2, self.latent_size)
        self.sigma = nn.Linear(32 * (self.input_size // 2**4)**2, self.latent_size)

        # Decoder
        self.decoder_linear = nn.Linear(self.latent_size, 32 * (self.input_size//2**4)**2)
        self.decoder = torch.nn.ModuleList([
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        ])

    def forward(self, x):
        # Encoder
        z, mu, log_sigma_sq = self.encoder(x)
        # Decoder
        z = self.decode(z)
        return z, mu, log_sigma_sq

    @torch.no_grad
    def sample(self, n):
        z = torch.randn(n, self.latent_size).to(self.decoder_linear.weight.device)
        z = self.decoder_linear(z).reshape([z.shape[0], 32, self.input_size//2**4, self.input_size//2**4])
        for module in self.decoder:
            z = module(z)
        return z

    def encode(self, x):
        for module in self.encoder:
            x = module(x)
        mu = self.mu(x.flatten(start_dim=1))
        log_sigma_sq = self.sigma(x.flatten(start_dim=1))
        sigma = torch.exp(0.5 * log_sigma_sq)

        # Reparametrization Trick
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z, mu, log_sigma_sq

    def decode(self, z):
        z = self.decoder_linear(z).reshape([z.shape[0], 32, self.input_size // 2 ** 4, self.input_size // 2 ** 4])
        for module in self.decoder:
            z = module(z)
        return z
