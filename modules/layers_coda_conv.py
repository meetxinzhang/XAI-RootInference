import torch
from torch import nn

import numpy as np
from torch.nn import functional as F


def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


class Im2Col(torch.autograd.Function):
    """
    This class does the same as torch.nn.functional.unfold(), but also implements the backward function.
    This allows for regularising the gradient by taking the second derivative.
    """
    @staticmethod
    def forward(ctx, x, kernel_size, padding, stride):
        ctx.shape, ctx.kernel_size, ctx.padding, ctx.stride = (x.shape[2:], kernel_size, padding, stride)
        return F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            shape, ks, padding, stride = ctx.shape, ctx.kernel_size, ctx.padding, ctx.stride
            return (
                F.fold(grad_output, shape, kernel_size=ks, padding=padding, stride=stride),
                None,
                None,
                None
            )


class Col2Im(torch.autograd.Function):
    """
    This class does the same as torch.nn.functional.fold(), but also implements the backward function.
    This allows for regularising the gradient by taking the second derivative.
    """
    @staticmethod
    def forward(ctx, x, shape, kernel_size, padding, stride):
        ctx.kernel_size, ctx.padding, ctx.stride = (kernel_size, padding, stride)
        return F.fold(x, shape, kernel_size=kernel_size, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            ks, padding, stride = ctx.kernel_size, ctx.padding, ctx.stride
            return (
                F.unfold(grad_output, kernel_size=ks, padding=padding, stride=stride),
                None,
                None,
                None,
                None
            )


def im2col(x, kernel_size, padding, stride):
    """
    Applies torch.nn.functional.unfold() to x with the given kernel size, padding and stride.
    Args:
        x: Input tensor to unfold.
        kernel_size: Kernel size of unfolding operation.
        padding: Padding used for the unfolding operation.
        stride: Stride used for the unfolding operation.

    Returns: The unfolded tensor as if F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        had been used.

    """
    return Im2Col.apply(x, kernel_size, padding, stride)


def col2im(x, shape, kernel_size, padding, stride):
    """
    Applies torch.nn.functional.fold() to x with the given kernel size, shape, padding and stride.
    Args:
        x: Input tensor to fold.
        shape: Shape of the output after folding.
        kernel_size: Kernel size of folding operation.
        padding: Padding used for the folding operation.
        stride: Stride used for the folding operation.

    Returns: The unfolded tensor as if F.fold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        had been used.

    """
    return Col2Im.apply(x, shape, kernel_size, padding, stride)


class AdaptiveSumPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, only that the normalisation is undone.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor) * np.prod(self.shape)


class FinalLayer(nn.Module):

    def __init__(self, norm=1, bias=-5):
        """
        Used to add a bias and a temperature scaling to the final output of a particular model.
        Args:
            norm: inverse temperature, i.e., T^{-1}
            bias: constant shift in logit space for all classes.
        """
        super().__init__()
        assert norm != 0, "Norm 0 means average pooling in the last layer of the old trainer. " \
                          "Please add size.prod() of final layer as img_size_norm to exp_params."
        self.norm = norm
        self.bias = bias

    def forward(self, in_tensor):
        out = (in_tensor.view(*in_tensor.shape[:2])/self.norm + self.bias)
        return out


class CoDAConv2d(nn.Module):

    def __init__(self, weighting_module):
        """
        Base implementation of a DAU convolution in 2D.
        Args:
            weighting_module: The module which predicts the matrix to be applied to the input.
                Moreover, it should have an .unfold(tensor) and a .fold(tensor) operation,
                which extracts patches from the input that are to be weighted by the weighting module.

        """
        super().__init__()
        self.weight_mod = weighting_module
        assert hasattr(self.weight_mod, "fold"), "The weighting module needs to implement a folding function."
        assert hasattr(self.weight_mod, "unfold"), "The weighting module needs to implement an unfolding function."

        # weight_matrix stores the last linear transformation that was applied to the input tensor of this module.
        # When applying regularisation to the combined matrix (see paper), this allows to map this matrix back
        # to the input for multiple targets.
        self.weight_matrix = None

        # If self.explain, self.activations will store the activation map of the last input tensor.
        # This can be useful when explaining intermediate activations via the gradient.
        self.explain = False
        self.activations = None

        # If self.reuse, the weightings will not be computed again. Instead, the last self.weight_matrix will be used.
        # This can be more efficient when evaluating the same input several times (e.g., analysing intermediate layers).
        self.reuse = False

        # If self.detach, self.weight_matrix.detach() is called before weighting the input. This allows for
        # obtaining the final linear weighting of the input for a given output by a simple backward pass.
        self.detach = False

        # Need this to be True if final linear mapping is regularised.
        self.save_matrix = False

    def forward(self, in_tensor):

        weightings, norm = self.weight_mod(in_tensor)
        unfolded_input = self.weight_mod.unfold(in_tensor, weightings.shape)

        if self.detach:
            # Can be used to calculate linear mappings between layers by only using the gradient.
            weightings = weightings.detach()
            norm = norm.detach()

        if self.save_matrix:
            # Save weights
            self.weight_matrix = weightings
        # Apply weight_matrix onto the input.
        activations = (unfolded_input * weightings)

        activations = activations.sum(1)

        activations = activations / norm.squeeze(1)

        return activations

    def explanation_mode(self, detach=True):
        """
        Enter 'explanation mode' by setting and self.detach.
        Args:
            detach: Whether or not to 'detach' the weight matrix from the computational graph so that it is not
                taken into account in the backward pass.

        Returns: None

        """
        self.detach = detach

    def reverse(self, back_tensor):
        """
                ONLY USED TO REGULARISE MATRICES.
        Reverse a dynamic linear convolution with the saved weight matrix.
        Args:
            back_tensor: Accumulated weighting up to this layer in the backwards matrix reconstruction.

        Returns: The 'back-projected' combined weight matrix.

        """
        # bsx is the number of classes to be explained times the batch size.
        bsx, _, h, w = back_tensor.shape
        bs = len(self.weight_matrix)
        # Reshape to (bs, #selected_classes, 1, *spatial_dims)
        # the 1-dim is used for broadcasting the weight that is placed on the sum to the individual summands
        # for the scalar products between matrix row and input vector.
        back_tensor = back_tensor.view(bs, -1, 1, *self.weight_matrix.shape[2:])
        back_tensor = (back_tensor * self.weight_matrix[:, None]).view(-1, *back_tensor.shape[2:])
        back_tensor = back_tensor.sum(2).view(bsx, -1, int(np.prod(back_tensor.shape[-2:])))
        return self.weight_mod.fold(back_tensor)


def select_classes(target, all_matrices, n):
    """
            ONLY USED TO REGULARISE MATRICES.
    Method for creating an indexing tensor to select a (sub)set of the classes to have their linear mapping reversed.

    Args:
        target: Ground truth target class as one-hot encoding.
        all_matrices (bool): If True, all matrices are selected.
        n: if not all_matrices, n - 1 classes are randomly chosen.

    Returns: an index tensor with (batch_size, n) of the n chosen classes per example.
        If all_matrices, then the standard order of the class indices is kept. Otherwise, the ground truth class
        will be the first in the list.

    """
    num_classes = target.shape[1]
    target = to_numpy(target)
    if all_matrices:
        return len(target) * [slice(None, None)]
    if n == 1:
        return target.argmax(1)[:, None]

        # Initialise matrix for the choices per sample
    out = np.zeros_like(to_numpy(target))

    if not (target.sum(1) > 1).any():
        # Let first entry always be the correct class.
        out[:, 0] = to_numpy(target.argmax(1))
    else:
        # Choose one of the possible entries for each batch item
        active_tgts = torch.stack(torch.where(target)).T
        for batch_idx in range(len(target)):
            sample_tgts = active_tgts[active_tgts[:, 0] == batch_idx, 1]
            out[batch_idx, 0] = sample_tgts[np.random.randint(len(sample_tgts))].item()

    out[:, 1:] = np.array([
        np.random.permutation(np.r_[0:t, t+1:num_classes]) for t in out[:, 0]
    ])
    return out


def reverse_linear_mappings(trainer, target, n_classes=2, all_matrices=False, level=None,
                            reshape=False, selection=None):
    """
            ONLY USED TO REGULARISE MATRICES.
    Traverses the model from output to input, reconstructing the overall linear transformation that
    took place. This assumes that the trainer.dlcn_layers are ordered correctly and do not have any non-linearity
    between them.
    Args:
        trainer: trainer object holding the model to be inverted.
        target: one-hot encoding of the ground truth classes.
        n_classes: number of linear transformations to be computed. For efficiency reasons not all matrices are computed
            during training. The order is always (correct class, n-1 randomly chosen classes).
        all_matrices: If True, compute all matrices in the standard order.
        level: If None, the linear transformation matrix w.r.t. the input is computed.
            Otherwise, level denotes the index of the last layer to be inversed, counting from
                output (0) of the network, to its input (level = # DLCNS).
        reshape: If True, the output is reshaped to (Batch size, Classes, *shape of input tensor).
            Otherwise, the last reshape is omitted and (Batch size x Classes, *shape of input tensor) is returned.
        selection: If not None, the given selection (list of class indices per batch example) is 'explained'.

    Returns: The matrices

    """
    dln_mods = trainer.dlcn_layers[::-1]
    shape = trainer.pool_layer.shape
    num_classes = trainer.options["num_classes"]
    if selection is None:
        all_idcs = select_classes(target, all_matrices, n_classes)
    else:
        all_idcs = selection
    # Undo the last pooling operation.
    if isinstance(trainer.pool_layer, AdaptiveSumPool2d):
        norm = 1
    else:
        norm = np.prod(shape)
    if trainer.final_layer is not None:
        norm *= trainer.final_layer.norm

    back_tensor = F.conv_transpose3d(torch.ones(1, 1, 1, *shape, dtype=torch.float).cuda() / norm,
                                     torch.eye(num_classes)[None, :, :, None, None].cuda())

    # Backproject the chosen idcs per sample.
    back_tensor = torch.cat([back_tensor[:, idcs] for idcs in all_idcs])
    back_tensor = back_tensor.reshape(-1, *back_tensor.shape[2:])

    if level is not None and level == 0:
        return back_tensor

    for lvl, mod in enumerate(dln_mods):
        back_tensor = mod.reverse(back_tensor)

        if level is not None and (lvl + 1) == level:
            return back_tensor
    if reshape:
        return back_tensor.view(len(all_idcs),  # This is supposed to be nothing more than the batch size
                                -1, *back_tensor.shape[1:])
    return back_tensor


class DAUWeights(nn.Module):

    def __init__(self, in_channels, num_daus, rank, norm_func,
                 kernel_size=3, stride=1, padding=None, groups=1, bias=True):

        """
        This module calculates the weights per position for a CoDA-ConvLayer.
        It consists mainly of a convolutional layer, which predicts the weightings that are to be applied to the input.

        The parameters kernel_size, stride, padding, groups, and bias are used for the nn.Conv2d.

        Args:
            in_channels: Number of input channels to convolution.
            num_daus: Number of output channels / DAUs in this layer.
            rank (int): The size of the rank to use for the DAUs.
            norm_func (nn.Module): Function to calculate the normalisation of the weights.
            kernel_size: Kernel size for the convolutional layer.
            stride: Stride for the convolutional layer.
            padding: Padding for the convolutional layer. If None, (KS-1)//2 is set for the padding.
            groups: Groups for the convolutional layer.
            bias: Bias for the convolutional layer.
        """
        super().__init__()
        if isinstance(rank, bool):
            assert not rank, "Bottleneck should be either False or the size of the bottleneck, not 'True'."

        self.shape = None
        self.out_channels = num_daus
        self.kernel_size = kernel_size
        if padding is None:
            padding = (kernel_size - 1)//2
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.weighting_f = nn.Sequential(
            nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=1, bias=False),
            nn.Conv2d(rank, num_daus * in_channels * kernel_size ** 2,
                      kernel_size=1, stride=1, padding=0, groups=groups, bias=bias)
        )
        self.norm_func = norm_func

    @staticmethod
    def reshape_out(in_tensor, out_c):
        """
        Reshapes the output into the format (B, input_patch_dim, out_c, H, W).
        Args:
            in_tensor: Tensor to be reshaped.
            out_c: The size of the out_c dimension. For the unfolded input, this will be one.
                When reshaping the weight matrix, this will be the number of output channels.

        Returns: The reshaped input tensor.

        """
        return in_tensor.view(in_tensor.shape[0],  # Batch size
                              -1,  # Input size to each convolutional kernel
                              out_c,  # Number of output channels
                              *in_tensor.shape[-2:])  # New height and width

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]  # Remember input spatial dimensions for folding operation.
        w = self.weighting_f(in_tensor)
        w = self.reshape_out(w, out_c=self.out_channels)
        norm = self.norm_func(w)
        return w, norm

    def unfold(self, in_tensor, shape):
        """
        This method is used to extract patches which are then weighted by the weighting_f.
        Args:
            in_tensor: Input tensor from which to extract patches.
            shape: shape of the weightings to easily extract correct shape.
        """

        return im2col(in_tensor, kernel_size=self.kernel_size,
                      padding=self.padding, stride=self.stride).view(
            in_tensor.shape[0],  # Batch size
            -1,  # Input size to each convolutional kernel
            1,  # Number of output channels
            *shape[-2:]  # New height and width
        )

    def fold(self, in_tensor):
        """
        This method is used in the backward pass to accumulate the weightings of the inputs.
        """
        return col2im(in_tensor, shape=self.shape, kernel_size=self.kernel_size,
                      padding=self.padding, stride=self.stride)