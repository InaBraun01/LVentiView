import torch
import torch.nn as nn
import torchvision.models as models


import sys
import torch
import torch.nn as nn
import torchvision.models as models

# PyTorch implementation of TernausNet16 with batch processing capability
class TernausNet16PyTorch_new(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(TernausNet16PyTorch, self).__init__()
        
        # Input conversion to 3 channels if needed
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
        else:
            self.input_conv = nn.Identity()
            
        # Load VGG16 as encoder (with pretrained weights)
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features)
        
        # Match Keras VGG16 structure for encoder
        self.enc_block1 = nn.Sequential(*features[:4])    # block1_conv1, block1_conv2
        self.enc_block2 = nn.Sequential(*features[5:9])   # block2_conv1, block2_conv2
        self.enc_block3 = nn.Sequential(*features[10:16])  # block3_conv1-3
        self.enc_block4 = nn.Sequential(*features[17:23]) # block4_conv1-3
        self.enc_block5 = nn.Sequential(*features[24:30]) # block5_conv1-3
        
        # Decoder blocks with parameters matching Keras model
        # First decoder block
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Second decoder block
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256+512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Third decoder block
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256+512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fourth decoder block
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fifth decoder block
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.final_conv = nn.Conv2d(32+64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(32, 4, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Process batch of images - input shape handling
        input_shape = x.shape
        
        # Handle input if it's in [batch, time_steps, H, W, C] format
        if len(input_shape) == 5:
            batch_size, time_steps, height, width, channels = input_shape
            # Flatten batch and time dimensions to process all frames at once
            x = x.view(-1, height, width, channels)
        
        # Handle NHWC to NCHW conversion (PyTorch uses channels first)
        if len(x.shape) == 4 and x.shape[3] == self.in_channels:
            # Convert from NHWC to NCHW
            x = x.permute(0, 3, 1, 2)

        # Input processing
        x = self.input_conv(x)

        # Encoder with skip connections
        skip1 = self.enc_block1(x)
        x = nn.functional.max_pool2d(skip1, kernel_size=2, stride=2, ceil_mode=True)
        
        skip2 = self.enc_block2(x)
        x = nn.functional.max_pool2d(skip2, kernel_size=2, stride=2, ceil_mode=True)
        
        skip3 = self.enc_block3(x)
        x = nn.functional.max_pool2d(skip3, kernel_size=2, stride=2)
        
        skip4 = self.enc_block4(x)
        x = nn.functional.max_pool2d(skip4, kernel_size=2, stride=2)
        
        skip5 = self.enc_block5(x)
        x = nn.functional.max_pool2d(skip5, kernel_size=2, stride=2)
        
        # Decoder with skip connections
        x = self.dec_conv1(x)
        x = torch.cat([x, skip5], dim=1)
        
        x = self.dec_conv2(x)
        x = torch.cat([x, skip4], dim=1)

        x = self.dec_conv3(x)
        x = torch.cat([x, skip3], dim=1)
        
        x = self.dec_conv4(x)
        x = torch.cat([x, skip2], dim=1)

        x = self.dec_conv5(x)
        x = torch.cat([x, skip1], dim=1)

        x = self.final_conv(x)
        x = self.relu(x)
        x = self.output_conv(x)
        x = self.softmax(x)
        
        # Output only the first 3 channels (matching Lambda layer in Keras)
        x = x[:, :3, :, :]
        
        # Reshape back to original batch structure if input was 5D
        if len(input_shape) == 5:
            x = x.view(batch_size, time_steps, 3, height, width)
        
        return x

# PyTorch implementation of TernausNet16 (does not handel batches)
class TernausNet16PyTorch(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(TernausNet16PyTorch, self).__init__()
        
        # Input conversion to 3 channels if needed
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
        else:
            self.input_conv = nn.Identity()
            
        # Load VGG16 as encoder (with pretrained weights)
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features)
        
        # Match Keras VGG16 structure for encoder
        self.enc_block1 = nn.Sequential(*features[:4])    # block1_conv1, block1_conv2
        self.enc_block2 = nn.Sequential(*features[5:9])   # block2_conv1, block2_conv2
        self.enc_block3 = nn.Sequential(*features[10:16])  # block3_conv1-3
        self.enc_block4 = nn.Sequential(*features[17:23]) # block4_conv1-3
        self.enc_block5 = nn.Sequential(*features[24:30]) # block5_conv1-3
        
        # Decoder blocks with parameters matching Keras model
        # First decoder block
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Second decoder block
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256+512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Third decoder block
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256+512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fourth decoder block
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fifth decoder block
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.final_conv = nn.Conv2d(32+64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(32, 4, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Handle NHWC to NCHW conversion (PyTorch uses channels first)

        if x.shape[1] != self.in_channels:
            # Assume NHWC and convert to NCHW
            x = x.permute(0, 3, 1, 2)

        # Input processing
        x = self.input_conv(x)

        # Encoder with skip connections
        skip1 = self.enc_block1(x)
        x = nn.functional.max_pool2d(skip1, kernel_size=2, stride=2,ceil_mode=True)
        
        skip2 = self.enc_block2(x)
        x = nn.functional.max_pool2d(skip2, kernel_size=2, stride=2,ceil_mode=True)
        
        skip3 = self.enc_block3(x)
        x = nn.functional.max_pool2d(skip3, kernel_size=2, stride=2)
        
        skip4 = self.enc_block4(x)
        x = nn.functional.max_pool2d(skip4, kernel_size=2, stride=2)
        
        skip5 = self.enc_block5(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        # Decoder with skip connections
        x = self.dec_conv1(x)
        x = torch.cat([x, skip5], dim=1)
        
        x = self.dec_conv2(x)
        x = torch.cat([x, skip4], dim=1)

        x = self.dec_conv3(x)
        x = torch.cat([x, skip3], dim=1)
        
        x = self.dec_conv4(x)
        x = torch.cat([x, skip2], dim=1)

        x = self.dec_conv5(x)
        x = torch.cat([x, skip1], dim=1)

        x = self.final_conv(x)
        x = self.relu(x)
        x = self.output_conv(x)
        x = self.softmax(x)
        
        # Output only the first 3 channels (matching Lambda layer in Keras)
        x = x[:, :3, :, :]
        
        return x



