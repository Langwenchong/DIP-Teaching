import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        encoder_channels = [8,16,32,64,128,256]
        for in_channels, out_channels in zip(encoder_channels[:-1], encoder_channels[1:]):
            self.encoder.add_module(
            f"conv_{in_channels}_{out_channels}",
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            self.encoder.add_module(
            f"batchnorm_{out_channels}",
            nn.BatchNorm2d(out_channels)
            )
            self.encoder.add_module(
            f"relu_{out_channels}",
            nn.ReLU(inplace=True)
            )
        
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

        decoder_channels = [128,64,32,16,8,3]
        self.decoder = nn.Sequential()
        for in_channels, out_channels in zip(decoder_channels[:-1], decoder_channels[1:]):
            self.decoder.add_module(
            f"conv_{in_channels}_{out_channels}",
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            self.decoder.add_module(
            f"batchnorm_{out_channels}",
            nn.BatchNorm2d(out_channels)
            )
            self.decoder.add_module(
            f"relu_{out_channels}",
            nn.ReLU(inplace=True)
            )
        self.output = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        x= self.encoder(x)
        # Decoder forward pass
        x = self.decoder(x)
        ### FILL: encoder-decoder forward pass
        output = self.output(x)
        
        return output
    