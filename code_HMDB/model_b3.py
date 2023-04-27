import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class LSTM_with_EFFICIENTNET(nn.Module):
    def __init__(self,num_classes,hidden_size, num_layers,pretrained,fine_tune):
        super(LSTM_with_EFFICIENTNET, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if pretrained:
            self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b3(weights=None)

        # modules=list(efficientnet.children())[:-1]
        # self.efficientnet=nn.Sequential(*modules)

        if fine_tune:
            for param in self.efficientnet.parameters():
                param.requires_grad = True
        else:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        

        self.lstm = nn.LSTM(1000, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
         # Reshape input tensor to [batch_size*frames, channels, height, width]
        batch_size = x.size(0)
        x = x.reshape(-1, 3, 224, 224)

        # Extract features using EfficientNet
        x = self.efficientnet(x)

        # Reshape output tensor to [batch_size, frames, features]
        x = x.view(batch_size, -1, x.size(-1))

        # Apply LSTM
        x, _ = self.lstm(x)

        # Get the last output from LSTM
        x = x[:, -1, :]

        # Apply fully connected layer
        x = self.linear(x)

        # Apply softmax activation
        x = self.softmax(x)

        return x 

