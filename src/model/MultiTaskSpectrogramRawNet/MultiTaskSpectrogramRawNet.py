import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np

from torch import Tensor
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter

from src.utils.spectrogram import Spectrogram, InverseSpectrogram


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


RESNET_CONFIGS = {
                  'recon': [[1, 1, 1, 1], PreActBlock],
                  '18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }


class Reconstruction_autoencoder(nn.Module):
    def __init__(self, enc_dim, nclasses=2):
        super(Reconstruction_autoencoder, self).__init__()

        self.fc = nn.Linear(enc_dim, 4 * 17 * 71)
        self.bn1 = nn.BatchNorm2d(4)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            PreActBlock(4, 16, 1),
            PreActBlock(16, 64, 1),
            PreActBlock(64, 128, 1)
        )

        self.layer2 = nn.Sequential(
            PreActBlock(128, 64, 1)
        )
        self.layer3 = nn.Sequential(
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4 = nn.Sequential(
            PreActBlock(16, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )

    def forward(self, z):
        z = self.fc(z).view((z.shape[0], 4, 17, 71))
        z = self.activation(self.bn1(z))
        z = self.layer1(z)
        z = self.layer2(z)
        z = nn.functional.interpolate(z, scale_factor=(3, 2), mode="bilinear", align_corners=True)
        z = self.layer3(z)
        z = nn.functional.interpolate(z, scale_factor=(3, 3), mode="bilinear", align_corners=True)
        z = self.layer4(z)

        return z


class compress_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(compress_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out



class compress_block(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride,*args, **kwargs):
        super(compress_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Conversion_autoencoder(nn.Module):
    def __init__(self, num_nodes, enc_dim, nclasses=2):
        self.in_planes = 16
        super(Conversion_autoencoder, self).__init__()

        self.spectrogram = Spectrogram()
        self.inverse_spectrogram = InverseSpectrogram()

        layers, block = RESNET_CONFIGS['recon']

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            compress_block(16, 16, 1),
            compress_block(16, 32, (2,3)),
        )

        self.layer2 = nn.Sequential(
            compress_block(32, 32, 1),
            compress_block(32, 64, 2),
        )

        self.layer3 = nn.Sequential(
            compress_block(64, 64, 1),
            compress_block(64, 128, 2),
        )

        self.layer4 = nn.Sequential(
            compress_block(128, 256, 1),
            compress_block(256, 128, 1),
        )

        # connect x_1

        self.layer1_i = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,2,1,output_padding=(0,1)),
            PreActBlock(256, 128, 1),
            PreActBlock(128, 64, 1),
        )

        self.layer2_i = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            PreActBlock(128, 64, 1),
            PreActBlock(64, 32, 1),
        )
        self.layer3_i = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, (2,3), 1,output_padding=(0,2)),
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4_i = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (9, 3), (3, 2), 1,output_padding=(2,1)),
            PreActBlock(32, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )



    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def forward(self, x):
        x, phase = self.spectrogram(x)

        x = self.conv1(x)
        x_1 = self.activation(self.bn1(x))
        x_2 = self.layer1(x_1)
        x_3 = self.layer2(x_2)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)
        y_1 = torch.cat([x_5,x_4],dim=1)
        y_2 = self.layer1_i(y_1)
        y_2 = torch.cat([y_2,x_3],dim=1)
        y_3 = self.layer2_i(y_2)
        y_3 = torch.cat([y_3,x_2],dim=1)
        y_4 = self.layer3_i(y_3)
        y_5 = torch.cat([y_4,x_1],dim=1)
        result = self.layer4_i(y_5)

        result = self.inverse_spectrogram(result, phase)

        return result


class Speaker_classifier(nn.Module):
    def __init__(self, enc_dim, nclasses):
        super(Speaker_classifier, self).__init__()
        self.fc_1 = nn.Linear(enc_dim, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, nclasses)
    def forward(self, x):
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        y = self.fc_3(x)
        return y


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(SincConv,self).__init__()

        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)
        fmel=self.to_mel(f)   # Hz to mel conversion
        fmelmax=np.max(fmel)
        fmelmin=np.min(fmel)
        filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
        filbandwidthsf=self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel=filbandwidthsf
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
        
    def forward(self,x):
        for i in range(len(self.mel)-1):
            fmin=self.mel[i]
            fmax=self.mel[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out


class RawNet(nn.Module):
    def __init__(self,
        nb_samp, first_conv, in_channels, filts, blocks, nb_fc_node, gru_node, nb_gru_layer, nb_classes, enc_dim,
        device):
        super(RawNet, self).__init__()
        
        self.device=device

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = filts[0],
			kernel_size = first_conv,
                        in_channels = in_channels
        )
        
        self.first_bn = nn.BatchNorm1d(num_features = filts[0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = filts[1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = filts[1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = filts[2]))
        filts[2][0] = filts[2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = filts[2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = filts[2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features = filts[1][-1],
            l_out_features = filts[1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features = filts[1][-1],
            l_out_features = filts[1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features = filts[2][-1],
            l_out_features = filts[2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features = filts[2][-1],
            l_out_features = filts[2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features = filts[2][-1],
            l_out_features = filts[2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features = filts[2][-1],
            l_out_features = filts[2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features = filts[2][-1])
        self.gru = nn.GRU(input_size = filts[2][-1],
			hidden_size = gru_node,
			num_layers = nb_gru_layer,
			batch_first = True)

        self.fc1_gru = nn.Linear(in_features = gru_node,
			out_features = nb_fc_node)

        self.fc1_feats = nn.Linear(in_features = gru_node,
            out_features = enc_dim)
       
        self.fc2_gru = nn.Linear(in_features = nb_fc_node,
			out_features = nb_classes,bias=True)
			
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, y = None):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x=x.view(nb_samp,1,len_seq)
        
        x = self.Sinc_conv(x)    
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x =  self.selu(x)
        
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1) # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1) # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1 # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1) # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2 # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1) # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3 # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1) # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4 # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1) # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5 # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]

        feats = self.fc1_feats(x)

        x = self.fc1_gru(x)
        logits = self.fc2_gru(x)
        # output=self.logsoftmax(x)
      
        return feats, logits

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        return nn.Sequential(*l_fc)

    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
        if print_fn == None: printfn = print
        model = self
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size
                        
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
                
            if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
                hooks.append(module.register_forward_hook(hook))
                
        device = device.lower()
        assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()
            
        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)


class MultiTaskSpectrogramRawNet(nn.Module):
    def __init__(self,
        nb_samp, first_conv, in_channels, filts, blocks, nb_fc_node, gru_node, nb_gru_layer, nb_classes, device,
        enc_dim,
        conv_ae_num_nodes,
        sc_nclasses,
        conv_ae_nclasses=2
    ):
        super().__init__()
        self.audio_model = RawNet(
            nb_samp, first_conv, in_channels, filts, blocks, nb_fc_node, gru_node, nb_gru_layer, nb_classes, enc_dim, device
        )
        
        self.reconstruction_autoencoder = Reconstruction_autoencoder(
            enc_dim=enc_dim
            )
        
        self.conversion_autoencoder = Conversion_autoencoder(
            num_nodes=conv_ae_num_nodes,
            enc_dim=enc_dim,
            nclasses=conv_ae_nclasses
        )

        self.speaker_classifier = Speaker_classifier(
            enc_dim=enc_dim,
            nclasses=sc_nclasses
        )
    