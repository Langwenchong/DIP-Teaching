import torch.nn as nn


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        encoder_channels = [3,32,64,128,256,512]
        for idx, (in_channels, out_channels) in enumerate(zip(encoder_channels[:-1], encoder_channels[1:])):
            self.add_module(
                f"en_conv{idx+1}",
                self.make_encoder_layers(in_channels, out_channels)
            )
        
        self.hidden_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

        decoder_channels = [512,256,128,64,32]
        for idx, (in_channels, out_channels) in enumerate(zip(decoder_channels[:-1], decoder_channels[1:])):
            self.add_module(
                f"de_conv{idx+1}",
                self.make_decoder_layers(in_channels, out_channels)
            )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def make_encoder_layers(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def make_decoder_layers(self,in_channels,out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.en_conv1(x) #[32,H/2,W/2]
        x2 = self.en_conv2(x1) #[64,H/4,W/4]
        X3 = self.en_conv3(x2) #[128,H/8,W/8]
        X4 = self.en_conv4(X3) #[256,H/16,W/16]
        X5 = self.en_conv5(X4) #[512,H/32,W/32]

        X6 = self.hidden_conv(X5)  #[512,H/32,W/32]
        X7 = self.hidden_conv(X6)   #[512,H/32,W/32]

        # Decoder forward pass
        output = self.de_conv1(X7) #[256,H/16,W/16]
        output = output+X4
        output = self.de_conv2(output) #[128,H/8,W/8]
        output = output+X3
        output = self.de_conv3(output) #[64,H/4,W/4]
        output = self.de_conv4(output) #[32,H/2,W/2]
        output = self.output(output) #[3,H,W]
        
        return output
    