import torch
import torch.nn as nn
import torch.nn.functional as func

class Encoder(nn.Module):
    def __init__(self, conf):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)  
        self.pool = nn.MaxPool2d(4, 4)

    def forward(self, x):
        x = func.relu(self.conv(x))
        x = self.pool(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, conf):
        super(Decoder, self).__init__()
        self.t_conv = nn.ConvTranspose2d(1, 3, 4, stride=4)

    def forward(self, x):
        x = func.relu(self.t_conv(x))
        x = torch.sigmoid(x)
        return x
        
class CAE(nn.Module):
    def __init__(self, conf):
        super(CAE, self).__init__()
        self.encoder = Encoder(conf)
        self.decoder = Decoder(conf)
        


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
                   
        return x