# -*- coding: utf-8 -*-
import torch

from models.resnet import BasicBlock, Bottleneck
from modules.piece_taylor import *
from modules.root_infer import *
from utils.visualization import visualize_featuremap


def taylor_2nd(Z, X, signal, S, log=''):
    dydx = torch.autograd.grad(Z, X, grad_outputs=torch.ones_like(Z), create_graph=True,
                               retain_graph=True)  # ([],)
    # 2ns gradient
    if dydx[0].requires_grad:
        # print(log, ', 2nd-Taylor Yes!')
        # C = self.gradprop(dydx[0], self.X, S)
        dy2dx = torch.autograd.grad(dydx[0], X, S, retain_graph=True)  # ([],)
    else:
        # C = dydx[0] * S
        dy2dx = [0]
    outputs = []

    a1 = signal * (dydx[0] * S)
    a2 = torch.pow(signal, exponent=2) * (dy2dx[0]) / 2
    outputs = a1 + a2
    return outputs


def model_flattening(module_tree):
    module_list = []
    children_list = list(module_tree.children())
    if len(children_list) == 0 or isinstance(module_tree, BasicBlock) or \
            isinstance(module_tree, Bottleneck):
        return [module_tree]
    else:
        for i in range(len(children_list)):
            module = model_flattening(children_list[i])
            module = [j for j in module]
            module_list.extend(module)
        return module_list


class ActivationStoringNet(nn.Module):
    def __init__(self, module_list):
        super(ActivationStoringNet, self).__init__()
        self.module_list = module_list

    def basic_block_forward(self, basic_block, activation):
        identity = activation

        basic_block.conv1.activation = activation
        activation = basic_block.conv1(activation)
        activation = basic_block.relu(basic_block.bn1(activation))
        basic_block.conv2.activation = activation
        activation = basic_block.conv2(activation)
        activation = basic_block.bn2(activation)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample)):
                basic_block.downsample[i].activation = identity
                identity = basic_block.downsample[i](identity)
            basic_block.identity = identity
        basic_block.activation = activation
        output = activation + identity
        output = basic_block.relu(output)

        return basic_block, output

    def bottleneck_forward(self, bottleneck, activation):
        identity = activation

        bottleneck.conv1.activation = activation
        activation = bottleneck.conv1(activation)
        activation = bottleneck.relu(bottleneck.bn1(activation))
        bottleneck.conv2.activation = activation
        activation = bottleneck.conv2(activation)
        activation = bottleneck.relu(bottleneck.bn2(activation))
        bottleneck.conv3.activation = activation
        activation = bottleneck.conv3(activation)
        activation = bottleneck.bn3(activation)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample)):
                bottleneck.downsample[i].activation = identity
                identity = bottleneck.downsample[i](identity)
            bottleneck.identity = identity
        bottleneck.activation = activation
        output = activation + identity
        output = bottleneck.relu(output)

        return bottleneck, output

    def forward(self, x):
        module_stack = []
        activation = x
        first_linear = True  # for vgg-16

        for i in range(len(self.module_list)):
            module = self.module_list[i]
            if isinstance(module, BasicBlock):
                module, activation = self.basic_block_forward(module, activation)
                module_stack.append(module)
            elif isinstance(module, Bottleneck):
                module, activation = self.bottleneck_forward(module, activation)
                module_stack.append(module)
            else:
                if isinstance(module, nn.Linear) and first_linear:  # for vgg-16
                    activation = activation.view(activation.size(0), -1)
                    first_linear = False
                module.activation = activation
                module_stack.append(module)
                activation = module(activation)
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    activation = activation.view(activation.size(0), -1)

        output = activation

        return module_stack, output


class DTDOpt(nn.Module):
    def __init__(self):
        super(DTDOpt, self).__init__()
        self.signal_map_1 = None
        self.signal_map_3 = None
        self.signal_map_5 = None
        self.signal_map_7 = None
        self.signal_map_9 = None
        self.signal_map_11 = None
        self.root_map = None

    def get_sn_maps(self):
        self.signal_map_7 = visualize_featuremap(self.signal_map_7)
        self.root_map = visualize_featuremap(self.root_map)
        return self.signal_map_7, self.root_map

    def get_noise(self):
        return self.root_map

    def get_signals(self):
        self.signal_map_1 = visualize_featuremap(self.signal_map_1)
        self.signal_map_3 = visualize_featuremap(self.signal_map_3)
        self.signal_map_5 = visualize_featuremap(self.signal_map_5)
        self.signal_map_7 = visualize_featuremap(self.signal_map_7)
        self.signal_map_9 = visualize_featuremap(self.signal_map_9)
        self.signal_map_11 = visualize_featuremap(self.signal_map_11)
        return (self.signal_map_1, self.signal_map_3, self.signal_map_5,
                self.signal_map_7, self.signal_map_9, self.signal_map_11)

    def forward(self, module_stack, y, class_num, model_archi, index=None):
        if index is None:
            R = torch.eye(class_num)[torch.max(y, 1)[1]].to(y.device)
        else:
            R = torch.eye(class_num)[index].to(y.device)
        R = torch.abs(R*y)
        # R = torch.abs(R)
        # r_layer = None
        for i in range(len(module_stack)):
            module = module_stack.pop()
            if len(module_stack) == 0:
                if isinstance(module, nn.Linear):
                    activation = module.activation
                    R = self.backprop_dense_input(activation, module, R)
                    print('last linear')
                elif isinstance(module, nn.Conv2d):
                    activation = module.activation
                    R = self.backprop_conv_input(activation, module, R)
                else:
                    raise RuntimeError(f'{type(module)} layer is invalid initial layer type')
            elif isinstance(module, BasicBlock):
                R = self.basic_block_R_calculate(module, R)
            elif isinstance(module, Bottleneck):
                # print('bottleneck...')
                R = self.bottleneck_R_calculate(module, R)
            else:
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    if model_archi == 'vgg':
                        R = R.view(R.size(0), -1, 7, 7)
                        continue
                    elif model_archi == 'resnet':
                        R = R.view(R.size(0), R.size(1), 1, 1)
                activation = module.activation
                R = self.R_calculate(activation, module, R)
        return R

    def basic_block_R_calculate(self, basic_block, R):
        if basic_block.downsample is not None:
            identity = basic_block.identity
        else:
            identity = basic_block.conv1.activation

        activation = basic_block.activation
        (R0, R1) = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(basic_block.conv2.activation, basic_block.conv2, R0,
                                         layer_idx=basic_block.layer_idx * 2 + 1)
        R0 = self.backprop_conv(basic_block.conv1.activation, basic_block.conv1, R0,
                                         layer_idx=basic_block.layer_idx * 2)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample) - 1, -1, -1):
                R1 = self.R_calculate(basic_block.downsample[i].activation,
                                               basic_block.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)
        return R

    def bottleneck_R_calculate(self, bottleneck, R):
        # print('bottleneck...')
        if bottleneck.downsample is not None:
            identity = bottleneck.identity
        else:
            identity = bottleneck.conv1.activation
        activation = bottleneck.activation
        R0, R1 = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(bottleneck.conv3.activation, bottleneck.conv3, R0)
        R0 = self.backprop_conv(bottleneck.conv2.activation, bottleneck.conv2, R0)
        R0 = self.backprop_conv(bottleneck.conv1.activation, bottleneck.conv1, R0)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample) - 1, -1, -1):
                R1 = self.R_calculate(bottleneck.downsample[i].activation,
                                      bottleneck.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)
        return R

    def R_calculate(self, activation, module, R):
        if isinstance(module, nn.Linear):
            # print('linear')
            R = self.backprop_dense(activation, module, R)
            return R
        elif isinstance(module, nn.Conv2d):
            R = self.backprop_conv(activation, module, R)
            return R
        elif isinstance(module, nn.BatchNorm2d):
            R = self.backprop_bn(R)
            return R
        elif isinstance(module, nn.ReLU):
            R = self.backprop_relu(activation, R)
            return R
        elif isinstance(module, nn.MaxPool2d):
            R = self.backprop_max_pool(activation, module, R)
            return R
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            R = self.backprop_adap_avg_pool(activation, R)
            return R
        elif isinstance(module, nn.Dropout):
            R = self.backprop_dropout(R)
            return R
        else:
            raise RuntimeError(f"{type(module)} can not handled currently")

    def backprop_dense_input(self, activation, module, R):
        pass
        return R

# --------------------------------经常更改的部分----------------------------------------------------------
    def backprop_dense(self, activation, module, R):
        # wp = torch.clamp(module.weight, min=0)
        # wn = torch.clamp(module.weight, max=0)
        #
        # root = rel_sup_root_linear(activation, R, step=1, weight=wp)
        # signal = activation - root
        #
        # zp = F.linear(signal, wp)  #
        # zn = F.linear(signal, wn)  #
        # Rp = piece_dtd_linear(x=signal, w=wp, z=zp, under_R=zp, R=R, root_zero=root, step=1)
        # Rn = piece_dtd_linear(x=signal, w=wn, z=zn, under_R=zn, R=R, root_zero=root, step=1)
        # R = Rp + Rn
        pw = torch.clamp(module.weight, min=0)
        nw = torch.clamp(module.weight, max=0)
        px = torch.clamp(activation, min=0)
        nx = torch.clamp(activation, max=0)

        def f(w1, w2, x1, x2):
            _z1 = F.linear(x1, w1)
            _z2 = F.linear(x2, w2)
            _R1 = R * torch.abs(_z1) / (torch.abs(_z1) + torch.abs(_z2))
            _R2 = R * torch.abs(_z2) / (torch.abs(_z1) + torch.abs(_z2))
            S1 = safe_divide(_R1, _z1)
            S2 = safe_divide(_R2, _z2)

            root1 = rel_sup_root_linear(x1, _R1, step=1, weight=w1, z=_z1)
            signal1 = x1 - root1

            # z1 = F.linear(signal1, w1)
            # R1 = piece_dtd_linear(x=signal1, w=w1, z=z1, under_R=z1, R=_R1, root_zero=root1, step=1)
            R1 = signal1 * torch.autograd.grad(_z1, x1, S1)[0]
            # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear

            root2 = rel_sup_root_linear(x2, _R2, step=5, weight=w2, z=_z2)
            signal2 = x2 - root2
            # z2 = F.linear(signal2, w2)
            # R2 = piece_dtd_linear(x=signal2, w=w2, z=_z2, under_R=z2, R=_R2, root_zero=root2, step=50)
            R2 = signal2 * torch.autograd.grad(_z2, x2, S2)[0]
            # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear
            return R1 + R2

        activator_relevances = f(pw, nw, px, nx)
        # inhibitor_relevances = f(nw, pw, px, nx)
        R = activator_relevances
        return R

    def backprop_conv(self, activation, module, R, layer_idx=9):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        wp = torch.clamp(module.weight, min=0)
        # wn = torch.clamp(module.weight, max=0)

        zp = F.conv2d(activation, wp, stride=stride, padding=padding)
        # zn = F.conv2d(activation, wn, stride=stride, padding=padding)
        # _R1 = R * torch.abs(zp) / (torch.abs(zp) + torch.abs(zn))
        # _R2 = R * torch.abs(zn) / (torch.abs(zp) + torch.abs(zn))

        root = rel_sup_root_cnn(activation, R, the_layer=[wp, stride, padding], step=5, z=zp)
        signal = activation - root

        # Sp = safe_divide(_R1, zp)
        # Sn = safe_divide(_R2, zn)

        Rp = signal * torch.autograd.grad(zp, activation, R)[0]
        # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear
        # Rn = signal * torch.autograd.grad(zn, activation, Sn)[0]

        R = Rp

        # xp = torch.clamp(activation, min=0)
        # xn = torch.clamp(activation, max=0)
        # print(xn.sum(dim=-1), 'all=0')

        # def f(w1, w2, x1, x2):
        #     _z1 = F.conv2d(x1, w1, bias=None, stride=stride, padding=padding)
        #     _z2 = F.conv2d(x2, w2, bias=None, stride=stride, padding=padding)
        #     _R1 = R * _z1 / (_z1 + _z2)
        #     _R2 = R * _z2 / (_z1 + _z2)
        #
        #     root1 = rel_sup_root_cnn(x1, _R1, the_layer=[w1, stride, padding])
        #     signal1 = x1 - root1
        #     z1 = F.conv2d(signal1, w1, bias=None, stride=stride, padding=padding)
        #     C1 = piece_dtd_conv(x=signal1, w=w1, z=z1, under_R=z1, R=_R1, root_zero=root1,
        #                         stride=stride, padding=padding)
        #
        #     root2 = rel_sup_root_cnn(x2, _R2, the_layer=[w2, stride, padding])
        #     signal2 = x2 - root2
        #     z2 = F.conv2d(signal2, w2, bias=None, stride=stride, padding=padding)
        #     C2 = piece_dtd_conv(x=signal2, w=w2, z=z2, under_R=z2, R=_R2, root_zero=root2,
        #                         stride=stride, padding=padding)
        #     return C2
        # activator_relevances = f(wp, wn, xp, xn)
        # # inhibitor_relevances = f(wn, wp, xp, xn)
        # R = 0.5 * activator_relevances  # + 0. * inhibitor_relevances

        # def f(w1, w2, x1, x2):
        #     Z1 = F.conv2d(x1, w1, bias=None, stride=stride, padding=padding)
        #     Z2 = F.conv2d(x2, w2, bias=None, stride=stride, padding=padding)
        #     S1 = safe_divide(R, Z1)
        #     S2 = safe_divide(R, Z2)
        #     C1 = x1 * torch.autograd.grad(Z1, x1, S1, retain_graph=True)[0]
        #     C2 = x2 * torch.autograd.grad(Z2, x2, S2, retain_graph=True)[0]
        #     return C1 + C2
        # activator_relevances = f(wp, wn, xp, xn)
        # inhibitor_relevances = f(wn, wp, xp, xn)
        # R = 1. * activator_relevances - 0. * inhibitor_relevances

        # if layer_idx == 8:
        #     self.signal_map_7 = R
        #     # self.root_map = root_p
        # if layer_idx == 1:
        #     self.signal_map_1 = R
        # if layer_idx == 3:
        #     self.signal_map_3 = R
        # if layer_idx == 5:
        #     self.signal_map_5 = R
        # if layer_idx == 7:
        #     self.signal_map_7 = R
        # if layer_idx == 9:
        #     self.signal_map_9 = R
        # if layer_idx == 11:
        #     self.signal_map_11 = R
        return R

    def backprop_conv_input(self, x, module, R):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        wp = torch.clamp(module.weight, min=0)
        # wn = torch.clamp(module.weight, max=0)
        x = torch.ones_like(x, dtype=x.dtype, requires_grad=True)

        zp = F.conv2d(x, wp, stride=stride, padding=padding)
        # zn = F.conv2d(x, wn, stride=stride, padding=padding)
        # _R1 = R * torch.abs(zp) / (torch.abs(zp) + torch.abs(zn))
        # _R2 = R * torch.abs(zn) / (torch.abs(zp) + torch.abs(zn))

        root = rel_sup_root_cnn(x, R, the_layer=[wp, stride, padding], step=5, z=zp)
        signal = x - root

        # Sp = safe_divide(_R1, zp)
        # Sn = safe_divide(_R2, zn)

        # Rp = piece_dtd_conv(signal, wp, stride, padding, z=zp, under_R=zp, R=_R1, root_zero=root, step=50)
        # Rn = piece_dtd_conv(signal, wn, stride, padding, z=zn, under_R=zn, R=_R2, root_zero=root, step=50)
        Rp = signal * torch.autograd.grad(zp, x, R)[0]
        # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear
        # Rn = signal * torch.autograd.grad(zn, x, Sn)[0]

        R = Rp
        # f(N) eval
        # self.root_map = root
        return R

    def backprop_bn(self, R):
        return R

    def backprop_dropout(self, R):
        return R

    def backprop_relu(self, activation, R):
        xp = torch.clamp(activation, min=0)
        xn = torch.clamp(activation, max=0)
        _z1 = F.relu(xp)
        _z2 = F.relu(xn)

        root1 = rel_sup_root_act(xp, R, step=20, func=F.relu, z=_z1)
        signal1 = activation - root1
        # z1 = F.gelu(xp)
        # R1 = piece_dtd_act(x=signal1, z=z1, under_R=z1, R=R, root_zero=root1, func=F.gelu, step=50)
        S1 = safe_divide(R, _z1)
        # R1 = signal1 * torch.autograd.grad(z1, xp, S1)[0]
        R1 = taylor_2nd(Z=_z1, X=xp, signal=signal1, S=S1)

        root2 = rel_sup_root_act(xn, R, step=20, func=F.relu, z=_z2)
        signal2 = xn - root2
        # z2 = F.gelu(xn)
        # R2 = piece_dtd_act(x=signal2, z=z2, under_R=z2, R=R, root_zero=root2, func=F.gelu, step=50)
        S2 = safe_divide(R, _z2)
        # R2 = signal2 * torch.autograd.grad(z2, xn, S2)[0]
        R2 = taylor_2nd(Z=_z2, X=xn, signal=signal2, S=S2)

        R = R1 + R2
        return R

    def backprop_adap_avg_pool(self, activation, R):
        kernel_size = activation.shape[-2:]
        Z = F.avg_pool2d(activation, kernel_size=kernel_size) * kernel_size[0] ** 2 + 1e-9
        S = R / Z
        R = activation * S
        return R

    def backprop_max_pool(sef, activation, module, R, ups=None):
        kernel_size, stride, padding = module.kernel_size, module.stride, module.padding
        Z, indices = F.max_pool2d(activation, kernel_size=kernel_size, stride=stride, \
                                  padding=padding, return_indices=True)
        Z = Z + 1e-9
        try:
            S = R / Z
            C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, \
                               padding=padding, output_size=activation.shape)
            R = activation * C
        except RuntimeError:
            R = R.view(R.size(0), -1, 7, 7)
            S = R / Z
            C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_size=activation.shape)
            R = activation * C
        return R

    def backprop_divide(self, R0, R1, ups=None):
        return R0 + R1

    def backprop_skip_connect(self, activation0, activation1, R, ups=None):
        Z = activation0 + activation1 + 1e-9
        S = R / Z
        R0 = activation0 * S
        R1 = activation1 * S
        return (R0, R1)
