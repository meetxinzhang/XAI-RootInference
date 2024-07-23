# -*- coding: utf-8 -*-
"""
 @time: 2023/10/11 20:37
 @desc: Forked from https://github.com/myc159/Deep-Taylor-Decomposition/blob/master/model/saliency_mapping.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock, Bottleneck


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

        for i in range(len(self.module_list)):
            module = self.module_list[i]
            if isinstance(module, BasicBlock):
                module, activation = self.basic_block_forward(module, activation)
                module_stack.append(module)
            elif isinstance(module, Bottleneck):
                module, activation = self.bottleneck_forward(module, activation)
                module_stack.append(module)
            else:
                module.activation = activation
                module_stack.append(module)
                activation = module(activation)
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    activation = activation.view(activation.size(0), -1)

        output = activation

        return module_stack, output


class DTD(nn.Module):
    def __init__(self, lowest=0., highest=255.):
        super(DTD, self).__init__()
        self.lowest = lowest
        self.highest = highest

    def forward(self, module_stack, y, class_num, model_archi):
        R = torch.eye(class_num)[torch.max(y, 1)[1]].to(y.device)

        for i in range(len(module_stack)):
            module = module_stack.pop()
            if len(module_stack) == 0:
                if isinstance(module, nn.Linear):
                    activation = module.activation
                    R = self.backprop_dense_input(activation, module, R)
                elif isinstance(module, nn.Conv2d):
                    activation = module.activation
                    R = self.backprop_conv_input(activation, module, R)
                else:
                    raise RuntimeError(f'{type(module)} layer is invalid initial layer type')
            elif isinstance(module, BasicBlock):
                R = self.basic_block_R_calculate(module, R)
            elif isinstance(module, Bottleneck):
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
        R0, R1 = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(basic_block.conv2.activation, basic_block.conv2, R0, layer_idx=basic_block.layer_idx*2+1)
        R0 = self.backprop_conv(basic_block.conv1.activation, basic_block.conv1, R0, layer_idx=basic_block.layer_idx*2)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample)-1, -1, -1):
                R1 = self.R_calculate(basic_block.downsample[i].activation,
                                      basic_block.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)

        return R

    def bottleneck_R_calculate(self, bottleneck, R):
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
            for i in range(len(bottleneck.downsample)-1, -1, -1):
                R1 = self.R_calculate(bottleneck.downsample[i].activation,
                                      bottleneck.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)

        return R

    def R_calculate(self, activation, module, R):
        if isinstance(module, nn.Linear):
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
        W_L = torch.clamp(module.weight, min=0)
        W_H = torch.clamp(module.weight, max=0)

        L = torch.ones_like(activation, dtype=activation.dtype) * self.lowest
        H = torch.ones_like(activation, dtype=activation.dtype) * self.highest
        # activation = torch.ones_like(img, dtype=img.dtype, requires_grad=True) * img
        # L = activation * self.lowest
        # H = activation * self.highest

        Z_O = torch.mm(activation, torch.transpose(module.weight, 0, 1))
        Z_L = torch.mm(activation, torch.transpose(W_L, 0, 1))
        Z_H = torch.mm(activation, torch.transpose(W_H, 0, 1))

        Z = Z_O - Z_L - Z_H + 1e-9
        S = R / Z

        C_O = torch.mm(S, module.weight)
        C_L = torch.mm(S, W_L)
        C_H = torch.mm(S, W_H)
        R = activation * C_O - L * C_L - H * C_H
        # C_0 = dtd_linear_piecewise_sample(x=activation, w=module.weight, z=Z, under_R=Z, R=R, funcs=[F.linear])
        # C_L = dtd_linear_piecewise_sample(x=L, w=W_L, z=Z, under_R=Z, R=R, funcs=[F.linear])
        # C_H = dtd_linear_piecewise_sample(x=H, w=W_H, z=Z, under_R=Z, R=R, funcs=[F.linear])
        # R = C_0 - C_L - C_H
        return R

    def backprop_dense(self, activation, module, R):
        # # z+
        # W = torch.clamp(module.weight, min=0)
        # # LRP-0 and v
        # W = module.weight
        # gamma
        W = 0.5*torch.clamp(module.weight, min=0) + 1.*module.weight
        Z = torch.mm(activation, torch.transpose(W, 0, 1)) + 1e-9
        S = R / Z
        C = torch.mm(S, W)  # R/Z * grad
        R = activation * C

        # # Zb ----------
        # W_L = torch.clamp(module.weight, min=0)
        # W_H = torch.clamp(module.weight, max=0)
        # L = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * self.lowest
        # H = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * self.highest
        # Z_O = torch.mm(activation, torch.transpose(module.weight, 0, 1))
        # Z_L = torch.mm(L, torch.transpose(W_L, 0, 1))
        # Z_H = torch.mm(H, torch.transpose(W_H, 0, 1))
        # Z = Z_O - Z_L - Z_H + 1e-9
        # S = R / Z
        # C_0 = torch.mm(S, module.weight)
        # C_l = torch.mm(S, W_L)
        # C_h = torch.mm(S, W_H)
        # R = activation * C_0 - L * C_l - H * C_h
        return R

    def backprop_conv(self, activation, module, R, layer_idx=9):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        # output_padding = activation.size(2) - ((R.size(2) - 1) * stride[0] - 2 * padding[0] + kernel[0])
        # # Montavon et al. 2019. Layer-Wise Relevance Propagation: An Overview. For ResNet18:
        # if layer_idx >= 13:  # LRP-0
        #     Z = F.conv2d(activation, module.weight, stride=stride, padding=padding) + 1e-9
        #     S = R / Z
        #     C = F.conv_transpose2d(S, module.weight, stride=stride, padding=padding, output_padding=output_padding)
        # elif layer_idx <= 6:  # LRP-gamma
        #     W = 0.25*torch.clamp(module.weight, min=0) + 1.*module.weight
        #     Z = F.conv2d(activation, W, stride=stride, padding=padding) + 1e-9
        #     S = R / Z
        #     C = F.conv_transpose2d(S, W, stride=stride, padding=padding, output_padding=output_padding)
        # else:  # LRP-varepsilon
        #     Z = F.conv2d(activation, module.weight, stride=stride, padding=padding) + 0.25
        #     S = R / Z
        #     C = F.conv_transpose2d(S, module.weight, stride=stride, padding=padding, output_padding=output_padding)

        # # Z+
        # W = torch.clamp(module.weight, min=0)
        # # LRP-0 and v
        # W = module.weight
        # gamma
        W = 0.5 * torch.clamp(module.weight, min=0) + 1. * module.weight
        Z = F.conv2d(activation, W, stride=stride, padding=padding)
        S = R / Z
        C = torch.autograd.grad(Z, activation, S)[0]
        R = activation * C

        # # Zb ----------
        # W_L = torch.clamp(module.weight, min=0)
        # W_H = torch.clamp(module.weight, max=0)
        # L = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * self.lowest
        # H = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * self.highest
        # Z_O = F.conv2d(activation, module.weight, stride=stride, padding=padding)
        # Z_L = F.conv2d(L, W_L, stride=stride, padding=padding)
        # Z_H = F.conv2d(H, W_H, stride=stride, padding=padding)
        # Z = Z_O - Z_L - Z_H + 1e-9
        # S = R / Z
        # C_0 = F.conv_transpose2d(S, module.weight, stride=stride, padding=padding, output_padding=output_padding)
        # C_l = F.conv_transpose2d(S, W_L, stride=stride, padding=padding, output_padding=output_padding)
        # C_h = F.conv_transpose2d(S, W_H, stride=stride, padding=padding, output_padding=output_padding)
        # R = activation * C_0 - L * C_l - H * C_h
        return R

    def backprop_conv_input(self, activation, module, R):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        output_padding = activation.size(2) - ((R.size(2) - 1) * stride[0] - 2 * padding[0] + kernel[0])

        # Zb ----------
        W_L = torch.clamp(module.weight, min=0)
        W_H = torch.clamp(module.weight, max=0)
        L = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * 0.
        H = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * 255.
        Z_O = F.conv2d(activation, module.weight, stride=stride, padding=padding)
        Z_L = F.conv2d(L, W_L, stride=stride, padding=padding)
        Z_H = F.conv2d(H, W_H, stride=stride, padding=padding)
        Z = Z_O - Z_L - Z_H + 1e-9
        S = R / Z
        C_0 = F.conv_transpose2d(S, module.weight, stride=stride, padding=padding, output_padding=output_padding)
        C_l = F.conv_transpose2d(S, W_L, stride=stride, padding=padding, output_padding=output_padding)
        C_h = F.conv_transpose2d(S, W_H, stride=stride, padding=padding, output_padding=output_padding)
        R = activation * C_0 - L * C_l - H * C_h

        # activation = torch.ones_like(activation, dtype=activation.dtype, requires_grad=True) * activation
        # # # Z+
        # # W = torch.clamp(module.weight, min=0)
        # # # LRP-0 and v
        # # W = module.weight
        # # gamma
        # W = 0.25 * torch.clamp(module.weight, min=0) + 1. * module.weight
        # Z = F.conv2d(activation, W, stride=stride, padding=padding)
        # S = R / Z
        # C = F.conv_transpose2d(S, W, stride=stride, padding=padding, output_padding=output_padding)
        # # C = torch.autograd.grad(Z, activation, S)[0]
        # R = activation * C
        return R

    def backprop_bn(self, R):
        return R

    def backprop_dropout(self, R):
        return R

    def backprop_relu(self, activation, R):
        # Z = F.relu(activation)  #
        # R = dtd_act_piecewise_sample(activation, Z, Z, R, zero_point='relu', funcs=[F.relu])
        return R

    def backprop_adap_avg_pool(self, activation, R):
        kernel_size = activation.shape[-2:]
        Z = F.avg_pool2d(activation, kernel_size=kernel_size) * kernel_size[0] ** 2 + 1e-9
        S = R / Z
        R = activation * S

        return R

    def backprop_max_pool(sef, activation, module, R):
        kernel_size, stride, padding = module.kernel_size, module.stride, module.padding
        Z, indices = F.max_pool2d(activation, kernel_size=kernel_size, stride=stride, \
                                  padding=padding, return_indices=True)
        Z = Z + 1e-9
        S = R / Z
        C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, \
                            padding=padding, output_size=activation.shape)
        R = activation * C

        return R

    def backprop_divide(self, R0, R1):
        return R0 + R1

    def backprop_skip_connect(self, activation0, activation1, R):
        Z = activation0 + activation1 + 1e-9
        S = R / Z
        R0 = activation0 * S
        R1 = activation1 * S

        return (R0, R1)