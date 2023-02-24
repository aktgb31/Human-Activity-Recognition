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
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)

        if fine_tune:
            for param in self.efficientnet.parameters():
                param.requires_grad = True
        else:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        # Change the final classification head.
        self.efficientnet.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

        self.lstm = nn.LSTM(1280, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
         # Reshape input tensor to [batch_size*frames, channels, height, width]
        x = x.reshape(-1, 3, 224, 224)

        # Extract features using EfficientNet
        x = self.efficientnet(x)

        # Reshape output tensor to [batch_size, frames, features]
        x = x.view(x.size(0), -1, x.size(-1))

        # Apply LSTM
        out, _ = self.lstm(x)

        # Get the last output from LSTM
        out = out[:, -1, :]

        # Apply fully connected layer
        out = self.fc(out)

        # Apply softmax activation
        out = self.softmax(out)

        return out

