import torch
import torch.nn as nn

from torch.nn import init

### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                #nn.UpsamplingBilinear3d(scale_factor=2),
                nn.Upsample(scale_factor=2, mode = 'trilinear'),
                nn.Conv3d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)



class UNet(nn.Module):

    def __init__(self, in_channels=2, n_classes=1, feature_scale=2, is_deconv=False, is_batchnorm=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv1 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv3(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.sigmoid=nn.Sigmoid()
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #print(inputs.size())

        # assume inputSize is 512

        conv1 = self.conv1(inputs)           # 16*512^2
        maxpool1 = self.maxpool(conv1)       # 16*256^2

        conv2 = self.conv2(maxpool1)         # 32*256^2
        maxpool2 = self.maxpool(conv2)       # 32*128^2

        conv3 = self.conv3(maxpool2)         # 64*128^2
        maxpool3 = self.maxpool(conv3)       # 64*64^2

        conv4 = self.conv4(maxpool3)         # 128*64^2
        maxpool4 = self.maxpool(conv4)       # 128*32^2

        center = self.center(maxpool4)       # 256*32^2
        up4 = self.up_concat4(center,conv4)  # 128*64^2
        up3 = self.up_concat3(up4,conv3)     # 64*128^2
        up2 = self.up_concat2(up3,conv2)     # 32*256^2
        up1 = self.up_concat1(up2,conv1)     # 16*512^2

        final = self.final(up1)
        final=self.sigmoid(final)
        return final



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    net = UNet(feature_scale=2).to(device)
    from torchsummary import summary
    summary(net, (2, 64, 64, 64))
    #summary(net)


    # print('#### Test Case ###')
    # from torch.autograd import Variable
    # x = Variable(torch.rand(2,3,32,32,32)).cuda()
    # model = UNet(in_channels = x.size()[1]).cuda()

    # print(model)

    # param = count_param(model)
    # y = model(x)
    # print('Output shape:',y.shape)
    # print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
