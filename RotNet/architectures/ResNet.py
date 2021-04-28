import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs))

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)
        
        feat = F.avg_pool2d(feat, feat.size(3)).view(-1, self.nChannels)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)
    
class ResNetCIFAR(nn.Module):
    def __init__(self, opt):#num_layers=20, num_stem_conv=32, config=(16, 32, 64)):
        super(ResNetCIFAR, self).__init__()
        
        num_classes = opt['num_classes']
        num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 2
        use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True

        num_layers = 20
        num_stem_conv = 32
        config = (96,160,192)        
        blocks = []
        
        self.num_layers = 20
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_stem_conv,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_stem_conv),
            nn.ReLU(True)
        )
        blocks.append(self.head_conv)
        
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = num_stem_conv
        for i in range(len(config)):
            for j in range(num_layers_per_stage):
                if j == 0 and i != 0:
                    strides = 2
                else:
                    strides = 1
                self.body_op.append(ResNet_Block(num_inputs, config[i], strides))
                num_inputs = config[i]
        self.body_op = nn.Sequential(*self.body_op)
        blocks.append(self.body_op)        
        # global average pooling and classifier
        self.tail_op=[]
        self.tail_op.append(GlobalAveragePooling())
        self.tail_op.append(nn.Linear(config[-1], num_classes))
        self.tail_op = nn.Sequential(*self.tail_op)
        blocks.append(self.tail_op)
        
        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]
        assert(len(self.all_feat_names) == len(self._feature_blocks))


    def _parse_out_keys_arg(self, out_feat_keys):

    	# By default return the features of the last layer / module.
    	out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

    	if len(out_feat_keys) == 0:
    		raise ValueError('Empty list of output feature keys.')
    	for f, key in enumerate(out_feat_keys):
    		if key not in self.all_feat_names:
    			raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
    		elif key in out_feat_keys[:f]:
    			raise ValueError('Duplicate output feature key: {0}.'.format(key))


    	# Find the highest output feature in `out_feat_keys
    	max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

    	return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
    	"""Forward an image `x` through the network and return the asked output features.

    	Args:
    	  x: input image.
    	  out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

    	Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
    	"""
    	out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
    	out_feats = [None] * len(out_feat_keys)

    	feat = x
    	for f in range(max_out_feat+1):
    		feat = self._feature_blocks[f](feat)
    		key = self.all_feat_names[f]
    		if key in out_feat_keys:
    			out_feats[out_feat_keys.index(key)] = feat

    	out_feats = out_feats[0] if len(out_feats)==1 else out_feats

    	return out_feats


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()

def create_model(opt):
    return ResNetCIFAR(opt)

if __name__ == '__main__':
    size = 32
    opt = {'num_classes':4, 'num_stages': 5}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

    out = net(x, out_feat_keys=net.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.all_feat_names[f], out[f].size()))


    out = net(x)
    print('Final output: {0}'.format(out.size()))
