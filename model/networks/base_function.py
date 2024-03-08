import re
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import math
from model.op import upfirdn2d, conv2d_gradfix


######################################################################################
# base function for network structure
######################################################################################


# def init_weights(net, init_type='normal', gain=0.02):
#     """Get different initial method for the network weights"""
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, 0.02)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic feature map, hence the input dim of SPADE


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(ADAIN)
    elif norm_type == 'spade':
        norm_layer = functools.partial(SPADE, config_text='spadeinstance3x3')
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params / 1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def _freeze_flow(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """

    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()

        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv1, nonlinearity, nonlinearity, conv2)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2)

    def forward(self, x):
        out = self.model(x)
        return out


class NormalBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(NormalBlock, self).__init__()

        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv)

    def forward(self, x):
        out = self.model(x)
        return out


'''.............................begin...................................'''


def make_kernel_NTED(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample_NTED(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel_NTED(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


# class EqualConv2d_NTED(nn.Module):
#     def __init__(
#         self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
#     ):
#         super().__init__()
#
#         self.weight = nn.Parameter(
#             torch.randn(out_channel, in_channel, kernel_size, kernel_size)
#         )
#         self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
#
#         self.stride = stride
#         self.padding = padding
#
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channel))
#
#         else:
#             self.bias = None
#
#     def forward(self, input):
#         out = conv2d_gradfix.conv2d(
#             input,
#             self.weight * self.scale,
#             bias=self.bias,
#             stride=self.stride,
#             padding=self.padding,
#         )
#
#         return out
#
#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
#             f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
#         )
#


class ToRGB_NTED(nn.Module):
    def __init__(
            self,
            in_channel,
            upsample=True,
            blur_kernel=[1, 3, 3, 1],
            norm_layer=nn.BatchNorm2d,
            nonlinearity=nn.LeakyReLU(),
            use_spect=False,
            use_coord=False
    ):
        super().__init__()

        if upsample:
            self.upsample = Upsample_NTED(blur_kernel)
        # self.conv = EqualConv2d_NTED(in_channel, 3, 3, stride=1, padding=1)
        self.outconv_NTED = Output(in_channel, 3, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, input, skip=None):
        # out = self.conv(input)
        out = self.outconv_NTED(input)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class ResBlock_DPTN(nn.Module):
    """
    Define an Residual block for different types
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock_DPTN, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc),
                                       nonlinearity, self.conv2, )

        self.shortcut = nn.Sequential(self.bypass, )

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


'''.............................end...................................'''


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """

    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2, )

        if self.learnable_shortcut:
            bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass, )

    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out


class ResBlocks(nn.Module):
    """docstring for ResBlocks"""

    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d,
                 nonlinearity=nn.LeakyReLU(), learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model = []
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            for i in range(num_blocks - 2):
                self.model += [ResBlock(hidden_nc, hidden_nc, hidden_nc,
                                        norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(
            nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity,
                                       conv2, )

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2, )
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class Output(nn.Module):
    """
    Define the output layer
    """

    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)),
                                       self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


class Jump(nn.Module):
    """
    Define the output layer
    """

    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)),
                                       self.conv1)

    def forward(self, x):
        out = self.model(x)
        return out


class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(LinearBlock, self).__init__()
        use_bias = True

        self.fc = spectral_norm(nn.Linear(input_nc, output_nc, bias=use_bias), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.fc)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.fc)

    def forward(self, x):
        out = self.model(x)
        return out


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    这段代码的作用是从输入的图像中提取图像块，并以C维度的形式返回这些块。它使用滑动窗口方法在图像上进行操作，并根据指定的参数进行裁剪和填充。
    输入:

    images: 输入的图像张量，形状为[batch, channels, in_rows, in_cols]，表示批次中的图像数量、通道数、图像的行数和列数。
    ksizes: 滑动窗口的大小，形状为[ksize_rows, ksize_cols]，表示窗口在每个维度上的大小。
    strides: 滑动窗口的步长，形状为[stride_rows, stride_cols]，表示窗口在每个维度上的步长。
    rates: 空洞率，形状为[dilation_rows, dilation_cols]，表示在每个维度上窗口内的采样间隔。
    padding: 填充类型，可以是'same'或'valid'，表示填充的方式。
    输出:
    patches: 提取的图像块张量，形状为[N, Ckk, L]，其中N是批次中的图像数量，C是通道数，k是滑动窗口的大小，L是总的图像块数。

    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4  # 检查输入图像张量的维度是否为4，确保输入正确
    assert padding in ['same', 'valid']

    if padding == 'same':  # 指定的填充类型，对输入图像进行填充或裁剪。
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    # 使用unfold对象对填充后的图像进行滑动窗口操作，提取图像块，并将它们展开成一个形状为[N, Ckk, L]的张量
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


class TCAM(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=2, in_channel=512, norm_layer=nn.BatchNorm2d,
                 nonlinearity=nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(TCAM, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.get_mask = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.Sigmoid())

    def forward(self, f_ct, f_cs, f_xs, test=False):
        raw_int_f_ct = list(f_ct.size())  # b*c*h*w
        raw_int_f_cs = list(f_cs.size())  # b*c*h*w

        kernel = 2 * self.rate

        phi_s = extract_image_patches(f_xs, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride,
                                               self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # 调用view函数对phi_s张量进行形状变换
        phi_s = phi_s.view(raw_int_f_cs[0], raw_int_f_cs[1], kernel, kernel, -1)
        # 调用permute函数对phi_s张量进行维度置换
        phi_s = phi_s.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        # 使用torch.split函数将phi_s张量沿着第0维度（批次大小维度）进行拆分
        phi_s_groups = torch.split(phi_s, 1, dim=0)

        f_ct = F.interpolate(f_ct, scale_factor=1. / self.rate, mode='nearest')  # 插值操作
        f_cs = F.interpolate(f_cs, scale_factor=1. / self.rate, mode='nearest')
        int_f_ct = list(f_ct.size())
        int_f_cs = list(f_cs.size())
        f_ct_groups = torch.split(f_ct, 1, dim=0)

        omiga_s = extract_image_patches(f_cs, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')

        omiga_s = omiga_s.view(int_f_cs[0], int_f_cs[1], self.ksize, self.ksize, -1)
        omiga_s = omiga_s.permute(0, 4, 1, 2, 3)
        omiga_s_groups = torch.split(omiga_s, 1, dim=0)

        f_xt = []
        offsets = []

        for f_ct_i, omiga_s_i, phi_s_i in zip(f_ct_groups, omiga_s_groups, phi_s_groups):
            escape_NaN = torch.FloatTensor([1e-4]).cuda()
            omiga_s_i = omiga_s_i[0]
            max_omiga_s_i = torch.max(torch.sqrt(reduce_sum(torch.pow(omiga_s_i, 2),
                                                            axis=[1, 2, 3],
                                                            keepdim=True)),
                                      escape_NaN)
            omiga_s_i_normed = omiga_s_i / max_omiga_s_i

            f_ct_i = same_padding(f_ct_i, [self.ksize, self.ksize], [1, 1], [1, 1])

            f_xt_i = F.conv2d(f_ct_i, omiga_s_i_normed, stride=1)

            f_xt_i = f_xt_i.view(1, int_f_cs[2] * int_f_cs[3], int_f_ct[2], int_f_ct[3])

            f_xt_i = F.softmax(f_xt_i * 10, dim=1)

            if not test:
                offset = torch.argmax(f_xt_i, dim=1, keepdim=True)

                offset = torch.cat([offset % int_f_ct[3], offset // int_f_ct[3]], dim=1)

            # deconv for patch pasting
            omiga_s_i_center = phi_s_i[0]
            if self.rate == 1:
                f_xt_i = F.pad(f_xt_i, [0, 1, 0, 1])
            f_xt_i = F.conv_transpose2d(f_xt_i, omiga_s_i_center, stride=self.rate, padding=1) / 4.
            f_xt.append(f_xt_i)
            if not test:
                offsets.append(offset)

        f_xt = torch.cat(f_xt, dim=0)

        f_xt.contiguous().view(raw_int_f_ct)

        mask = self.get_mask(f_xt)

        flow = None
        grids = None
        if not test:
            offsets = torch.cat(offsets, dim=0)
            offsets = offsets.view(int_f_ct[0], 2, *int_f_ct[2:])

            grids = offsets.type(torch.FloatTensor)
            grids = grids.cuda()

            h_add = torch.arange(int_f_ct[2]).view([1, 1, int_f_ct[2], 1]).expand(int_f_ct[0], -1, -1, int_f_ct[3])
            w_add = torch.arange(int_f_ct[3]).view([1, 1, 1, int_f_ct[3]]).expand(int_f_ct[0], -1, int_f_ct[2], -1)
            ref_coordinate = torch.cat([h_add, w_add], dim=1)
            ref_coordinate = ref_coordinate.cuda()

            offsets = offsets - ref_coordinate
            flow = offsets.type(torch.FloatTensor).cuda()

        return f_xt, mask, flow, grids


class ResBlockEncoderOptimized(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoderOptimized, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(conv1, norm_layer(hidden_nc), nonlinearity, conv2, )
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class EncoderBlockOptimized(nn.Module):
    """
        Define an Encoder block for the first layer of the generator
    """

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlockOptimized, self).__init__()

        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv1, nonlinearity, conv2)
        else:
            self.model = nn.Sequential(conv1, norm_layer(output_nc), nonlinearity, conv2)

    def forward(self, x):
        out = self.model(x)
        return out