# -*- coding: utf-8 -*-
"""
 @time: 2024/1/11 21:12
 @desc:
"""
import torch

from modules.piece_taylor import *
from modules.root_infer import *


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, inputs, output):
    if type(inputs[0]) in (list, tuple):
        self.X = []
        for i in inputs[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = inputs[0].detach()
        self.X.requires_grad = True
    self.Y = output


class Interpretation(nn.Module):  # Deep TayLor Decomposition      326
    def __init__(self):
        super(Interpretation, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def relprop(self, R, alpha):
        return R


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


class Linear(nn.Linear, Interpretation):
    def relprop(self, R, alpha):
    #     root = rel_sup_root_linear(self.X, R, step=10, weight=self.weight)
    #     signal = self.X - root
    #     z = F.linear(self.X, self.weight)
    #     # R = piece_dtd_linear(x=signal, w=self.weight, z=z, under_R=z, R=R, root_zero=root)
    #     S = safe_divide(R, z)
    #     R = signal * torch.autograd.grad(z, signal, S)[0]
    #     # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            _z1 = F.linear(x1, w1)
            _z2 = F.linear(x2, w2)
            _R1 = R * torch.abs(_z1) / (torch.abs(_z1) + torch.abs(_z2))
            _R2 = R * torch.abs(_z2) / (torch.abs(_z1) + torch.abs(_z2))

            root1 = rel_sup_root_linear(x1, _R1, step=1, weight=w1)
            signal1 = x1 - root1
            # z1 = F.linear(x1, w1)
            # C1 = piece_dtd_linear(x=signal1, w=w1, z=z1, under_R=z1, R=R, root_zero=root1, step=1)
            S1 = safe_divide(_R1, _z1)
            C1 = signal1 * torch.autograd.grad(_z1, x1, S1)[0]
            # C1 = taylor_2nd(Z=z1, X=x1, signal=signal1, S=S1)

            root2 = rel_sup_root_linear(x2, _R2, step=50, weight=w2)
            signal2 = x2 - root2
            # z2 = F.linear(x2, w2)
            # C2 = piece_dtd_linear(x=signal2, w=w2, z=z2, under_R=z2, R=R, root_zero=root2, step=50)
            S2 = safe_divide(_R2, _z2)
            C2 = signal2 * torch.autograd.grad(_z2, x2, S2)[0]
            # C2 = taylor_2nd(Z=z2, X=x2, signal=signal2, S=S2)
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)
        R = 0.5*activator_relevances + 0.*inhibitor_relevances
        return R


class GELU(nn.GELU, Interpretation):
    def __init__(self, layer_idx=9):
        super(GELU, self).__init__()
        self.layer_idx = layer_idx

    # def relprop(self, R, alpha):
    #     z = F.gelu(self.X)
    #     root = rel_sup_root_act(self.X, R, step=20, func=F.softmax,  z=z)
    #     # root = torch.zeros_like(self.X)
    #     signal = self.X - root
    #     # R = piece_dtd_act(x=signal, z=z, under_R=z, R=R, root_zero=root, func=F.gelu, step=20)
    #     S = safe_divide(R, z)
    #     # R = signal * torch.autograd.grad(z, signal, S)[0]
    #     R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)

    def relprop(self, R, alpha):
        xp = torch.clamp(self.X, min=0)
        xn = torch.clamp(self.X, max=0)
        _z1 = F.gelu(xp)
        _z2 = F.gelu(xn)

        root1 = rel_sup_root_act(xp, R, step=50, func=F.gelu, z=_z1)
        signal1 = self.X - root1
        # z1 = F.gelu(xp)
        # R1 = piece_dtd_act(x=signal1, z=z1, under_R=z1, R=R, root_zero=root1, func=F.gelu, step=50)
        S1 = safe_divide(R, _z1)
        # R1 = signal1 * torch.autograd.grad(z1, xp, S1)[0]
        R1 = taylor_2nd(Z=_z1, X=xp, signal=signal1, S=S1)

        root2 = rel_sup_root_act(xn, R, step=50, func=F.gelu, z=_z2)
        signal2 = xn - root2
        # z2 = F.gelu(xn)
        # R2 = piece_dtd_act(x=signal2, z=z2, under_R=z2, R=R, root_zero=root2, func=F.gelu, step=50)
        S2 = safe_divide(R, _z2)
        # R2 = signal2 * torch.autograd.grad(z2, xn, S2)[0]
        R2 = taylor_2nd(Z=_z2, X=xn, signal=signal2, S=S2)

        R = 0.5*R1 + 0.*R2

        # ablation
        if self.layer_idx == 0:
            y1 = F.gelu(self.X)
            root_n = root.clone().detach().requires_grad_(True)
            y2 = F.gelu(root_n)
            grad_x = torch.autograd.grad(outputs=y1, inputs=self.X, grad_outputs=torch.ones_like(y1))[0]
            grad_r = torch.autograd.grad(outputs=y2, inputs=root_n, grad_outputs=torch.ones_like(y2))[0]
            var = F.mse_loss(grad_x, grad_r, reduce=True, size_average=True)
            print('MSE ', var)
        return R
    pass


class Softmax(nn.Softmax, Interpretation):
    def relprop(self, R, alpha):
        z = F.softmax(self.X, dim=-1)
        root = rel_sup_root_act(self.X, R, step=50, func=F.softmax, z=z)
        # root = torch.zeros_like(self.X)
        signal = self.X - root
        # R = piece_dtd_act(x=signal, z=z, under_R=z, R=R, root_zero=root, func=F.softmax, step=50)
        S = safe_divide(R, z)
        # R = signal * torch.autograd.grad(z, self.X, S)[0]
        R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)
        return R
        # px = torch.clamp(self.X, min=0)
        # nx = torch.clamp(self.X, max=0)
        #
        # root1 = rel_sup_root_act(px, R, step=100, func=F.softmax)
        # signal1 = px - root1
        # z1 = F.softmax(signal1, dim=-1)
        # R1 = piece_dtd_act(x=signal1, z=z1, under_R=z1, R=R, root_zero=root1, func=F.softmax, step=100)
        #
        # root2 = rel_sup_root_act(nx, R, step=100, func=F.softmax)
        # signal2 = nx - root2
        # z2 = F.softmax(signal2, dim=-1)
        # R2 = piece_dtd_act(x=signal2, z=z2, under_R=z2, R=R, root_zero=root2, func=F.softmax, step=100)
        # return 0.5*R1 + 0.1*R2
    pass


class LayerNorm(nn.LayerNorm, Interpretation):
    pass


class Dropout(nn.Dropout, Interpretation):
    pass


class LayerNorm(nn.LayerNorm, Interpretation):
    pass


class IndexSelect(Interpretation):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = torch.autograd.grad(Z, self.X, S, retain_graph=True)

        if not torch.is_tensor(self.X):
            outputs = [self.X[0] * C[0], self.X[1] * C[1]]
        else:
            outputs = self.X * (C[0])
        return outputs


class Clone(Interpretation):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = torch.autograd.grad(Z, self.X, S, retain_graph=True)[0]

        R = self.X * C

        return R


class Cat(Interpretation):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class BatchNorm2d(nn.BatchNorm2d, Interpretation):
    def relprop(self, R, alpha):
        X = self.X
        # beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Conv2d(nn.Conv2d, Interpretation):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R


class RelPropSimple(Interpretation):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = torch.autograd.grad(Z, self.X, S, retain_graph=True)

        if not torch.is_tensor(self.X):
            outputs = [self.X[0] * C[0], self.X[1] * C[1]]
        else:
            outputs = self.X * (C[0])
        return outputs


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = torch.autograd.grad(Z, self.X, S, retain_graph=True)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)


