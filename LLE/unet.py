import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, batch_norm= False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.bn1(x)   
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, batch_norm= batch_norm)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        
        p = self.pool(x)
        
        return x,p
    
class Decoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c, batch_norm=batch_norm)
    def forward(self, inputs, skip):
        
        x = self.up(inputs)
        
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, batch_norm = False):
        super().__init__()
        '''Encoder'''
        self.e1 = Encoder(in_c=in_channels, out_c = 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        
        '''bottle neck'''
        self.b = Encoder(512, 1024, batch_norm=batch_norm)
        
        '''Decoder'''
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        """ Bottleneck """
        b, _ = self.b(p4)
        
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs
        
        
        
        