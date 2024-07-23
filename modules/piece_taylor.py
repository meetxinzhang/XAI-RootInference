# -*- coding: utf-8 -*-
"""
 @time: 2024/1/17 16:24
 @desc:
"""
# import torch
# import torch.nn.functional as F

zero_points = {'gelu': 0.0, 'relu': 1e-9, 'softmax': -5, 'zero': 0.0}


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)  # set the min bound, means get larger than 1e-9, the "stabilizer"
    den = den + den.eq(0).type(den.type()) * 1e-9  # if den==0 then +1*1e-9
    return a / den * b.ne(0).type(b.type())  # do / if b=!0 or *0
#
#
# def root_routing_sample(x, root_t_s, step=10):
#     # b, t, e = x.size()  # [b t e]
#     if not torch.is_tensor(root_t_s):
#         root_zero = torch.zeros_like(x, requires_grad=True).to(x.device)
#         root_zero = root_zero + zero_points[root_t_s] + 1e-9
#     else:
#         root_zero = root_t_s
#
#     media_roots = []
#     delta = (x - root_zero) / step
#     for i in range(1, step):
#         media_roots.append(root_zero + delta * i)
#         # delta = (x - root_zero) / 5
#     # root1 = root_zero + delta
#     # root2 = root1 + delta
#     # root3 = root2 + delta
#     # root4 = root3 + delta
#     return delta, media_roots
#
#
# def piece_dtd_linear(x, w, z, under_R, R, root_zero, step=10):
#     # inn = torch.zeros_like(Z)  # [b token e_out]
#     # Z1 = F.linear(x1, w1)  # y=R=[b token e_out]
#     # Z2 = F.linear(x2, w2)  #
#     S = safe_divide(R, under_R)  # R/Zj
#     delta, media_roots = root_routing_sample(x, root_zero, step=step)
#
#     grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
#     relevance = delta * grad_x
#
#     for root in media_roots:
#         z = F.linear(root, w)
#         grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
#         relevance = relevance + delta * grad
#     return relevance
#
#
# def piece_dtd_conv(x, w, stride, padding, z, under_R, R, root_zero, step=10):
#     S = safe_divide(R, under_R)  # R/Zj
#     delta, media_roots = root_routing_sample(x, root_zero, step=step)
#     grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
#     relevance = delta * grad_x
#
#     for root in media_roots:
#         z = F.conv2d(root, w, bias=None, stride=stride, padding=padding)
#         grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
#         relevance = relevance + delta * grad
#     return relevance
#
#
# def piece_dtd_act(x, z, under_R, R, root_zero, func=None, step=5):
#     S = safe_divide(R, under_R)  # R/Zj
#     delta, media_roots = root_routing_sample(x, root_zero, step=step)
#
#     grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
#     relevance = delta * grad_x
#
#     for root in media_roots:
#         if func is F.softmax:
#             z = func(root, dim=-1)
#         else:
#             z = func(root)
#         grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
#         relevance = relevance + delta * grad
#     return relevance
#
#
# def piece_dtd_linear_v2(x, z, S, root_zero, func=None, step=10):
#     # inn = torch.zeros_like(Z)  # [b token e_out]
#     # Z1 = F.linear(x1, w1)  # y=R=[b token e_out]
#     # Z2 = F.linear(x2, w2)  #
#     # S = safe_divide(R, under_R)  # R/Zj
#     delta, media_roots = root_routing_sample(x, root_zero, step=step)
#
#     grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
#     relevance = delta * grad_x
#
#     for root in media_roots:
#         z = func(root)
#         grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
#         relevance = relevance + delta * grad
#     return relevance
#
#
# def piece_dtd_conv_v2(x, z, under_R, R, root_zero, func=None, step=10):
#     S = safe_divide(R, under_R)  # R/Zj
#     delta, media_roots = root_routing_sample(x, root_zero, step=step)
#     grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
#     relevance = delta * grad_x
#
#     for root in media_roots:
#         z = func(root)
#         grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
#         relevance = relevance + delta * grad
#     return relevance