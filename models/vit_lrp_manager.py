import argparse
import torch
import numpy as np
import cv2


# compute rollout between attention layers
# def compute_rollout_attention(all_layer_matrices, start_layer=0):
#     # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
#     num_tokens = all_layer_matrices[0].shape[1]
#     batch_size = all_layer_matrices[0].shape[0]
#     eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
#     all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
#     matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
#                     for i in range(len(all_layer_matrices))]
#     joint_attention = matrices_aug[start_layer]
#     for i in range(start_layer + 1, len(matrices_aug)):
#         joint_attention = matrices_aug[i].bmm(joint_attention)
#     return joint_attention


def ignite_relprop(model, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0,
                   alpha=1):
    model.eval()
    output = model(input)
    kwargs = {"alpha": alpha}
    if index is None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    # cam = model.relprop((torch.tensor(one_hot_vector).to(output.device) * output).to(input.device), method=method,
    #                     is_ablation=is_ablation,
    #                     start_layer=start_layer, **kwargs)
    cam = model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method,
                        is_ablation=is_ablation,
                        start_layer=start_layer, **kwargs)
    return cam_interpolate(cam)
    # return cam.unsqueeze(1)


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def cam_interpolate(cam):
    # cam [b, 196]
    cam = cam.squeeze().reshape(1, 1, 14, 14)  # [1, 196]->[1, 1, 14, 14]
    cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear')
    return cam


def visualize_attention(original_image, cam):
    # cam [1, 1, 224, 224]
    # original_image [3, 224, 224]
    original_image = original_image.squeeze()  # [1, 3, 224, 224]
    # cam = cam_interpolate(cam)
    cam = cam.reshape(224, 224).cuda().data.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min()) + 1e-9
    img = original_image.permute(1, 2, 0).data.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, cam)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def visualize_full_cam(original_image, cam):
    # cam = cam.reshape(1, 1, 14, 14)
    # cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear')
    cam = cam.reshape(224, 224).cuda().data.cpu().numpy()
    cam = (cam - cam.min()) / (
            cam.max() - cam.min())
    img = original_image.permute(1, 2, 0).data.cpu().numpy()
    img = (img - img.min()) / (
            img.max() - img.min())
    vis = show_cam_on_image(img, cam)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis
