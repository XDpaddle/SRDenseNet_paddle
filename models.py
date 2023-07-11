import x2paddle
from x2paddle import torch2paddle
import paddle
import math
from paddle import nn
from paddleseg.cvlibs import param_init

class ConvLayer(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=\
            kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=\
            kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return paddle.concat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Layer):

    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate,
                kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return paddle.concat([x, self.block(x)], 1)


class SRDenseNet(nn.Layer):

    def __init__(self, num_channels=1, growth_rate=16, num_blocks=8,
        num_layers=8):
        super(SRDenseNet, self).__init__()
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers *
                (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        self.bottleneck = nn.Sequential(nn.Conv2D(growth_rate * num_layers +
            growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.ReLU())
        self.deconv = nn.Sequential(torch2paddle.Conv2DTranspose(256, 256,
            kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.ReLU(), torch2paddle.Conv2DTranspose(256, 256, kernel_size=3, stride=2, padding=3 //
            2, output_padding=1), nn.ReLU())
        self.reconstruction = nn.Conv2D(256, num_channels, kernel_size=3,padding=3 // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, paddle.nn.Conv2D) or isinstance(m,torch2paddle.Conv2DTranspose):
                
                param_init.normal_init(m.weight.data,mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                if m.bias is not None:
                    param_init.constant_init(m.bias.data,value=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        x = self.deconv(x)
        x = self.reconstruction(x)
        return x
