# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
import numpy as np
import time
from models.resnet import resnet18 as resnet
# from models.resnet import resnet50_coco as resnet
# import torchvision
# from modules import layers_cnn as sa_opt
# from CNNs.baselines import dtd_ww as sa_base
# from CNNs.baselines import dtd_monvaton as sa_base
# from CNNs.baselines import lrp_0 as sa_base
# from CNNs.baselines import dtd_z_b as sa_base
from modules.baselines import dtd_z_plus as sa_base
# from models.vit import vit_base_patch16_224 as vit_lrp
# from models.vit_exp import vit_base_patch16_224 as vit_lrp_exp
# from models.vit_lrp_manager import ignite_relprop
# from models.vgg_AGF import vgg19
# from models.vgg import vgg19
# from data.voc import VOCSegmentation
# from data.imagenet import ImagenetSegDataset
from data.coco import COCOSegDataset
from tqdm import tqdm
import cv2
import einops


# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


def pixel_accuracy_batch(vis, label):  # [b 224 224]
    vis = vis.cpu().numpy()
    label = label.cpu().numpy()
    label = label.reshape(label.shape[0], -1)
    vis = vis.reshape(vis.shape[0], -1)
    pixel_labeled = np.sum(label > 0)
    if pixel_labeled == 0:
        return None
    pixel_correct = np.sum((vis == label) * (label > 0))
    return pixel_correct / pixel_labeled


def iou_batch(vis, label):  # [b 224 224]
    vis = vis.cpu().numpy()
    label = label.cpu().numpy()
    label = label.reshape(label.shape[0], -1)
    vis = vis.reshape(vis.shape[0], -1)
    intersection = np.sum((vis == label) * (label > 0))
    union = np.sum((vis + label) > 0) / 2
    return intersection / union


pixel_acc = []
intersection = []


def eval(model, loader):
    iterator = tqdm(loader)
    net = sa_base.ActivationStoringNet(sa_base.model_flattening(model)).cuda()
    DTD = sa_base.DTD().cuda()
    # net = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    # DTD = sa_opt.DTDOpt().cuda()

    for batch_idx, (x, mask, label) in enumerate(iterator):  # [B 3 224 224], [B 224 224]
        x = x.cuda()
        mask = mask.to(x.device)

        # model inference
        module_stack, output = net(x)
        dtd_start = time.time()
        # vis = DTD(module_stack, output, 1000, 'resnet')  # [b, 3, 224, 224]
        vis = DTD(module_stack, output, 1000, 'resnet', index=label)  # [b, 3, 224, 224]
        dtd_end = time.time()
        print("耗时: {:.2f}秒".format(dtd_end - dtd_start))
        # _ = model(x)
        # vis = ignite_relprop(model, input=x,  # [b 1 224 224]
        #                      method="transformer_attribution", alpha=1,
        #                      index=mask.data.cpu()).detach()
        # kwargs = {
        #     'no_a': False,
        #     'no_fx': False,
        #     'no_fdx': False,
        #     'no_m': False,
        #     'no_reg': False,
        #     'gradcam': False
        # }
        # vis = model.AGF(**kwargs)

        vivi = []
        for v in vis:
            v = torch.sum(v, dim=0, keepdim=False)
            v = (v - v.min()) / (v.max() - v.min())  # normalize
            ret = v.mean()
            v = v.gt(ret)
            v = torch.where(torch.gt(v, ret), torch.ones_like(v), v)
            v = v.cpu().data.numpy()
            vivi.append(v)
        vis = np.array(vivi)
        vis = torch.from_numpy(vis).to(x.device)

        # vivi = []
        # for v in vis:
        #     v = torch.sum(v, dim=0)
        #     # v = gaussian_filter(v.cpu().data.numpy(), sigma=1)
        #     v = (v - v.min()) / (v.max() - v.min() + 1e-9)  # normalize
        #     v = v.cpu().data
        #     v = cv2.applyColorMap(np.uint8(255 * v), cv2.COLORMAP_TURBO)  # TURBO 线性， JET 减弱为微小值
        #     v = np.float32(v) / 255
        #     vivi.append(v)
        # vis = np.array(vivi)
        # vis = torch.from_numpy(vis).to(x.device)
        # vis = einops.rearrange(vis, 'b h w c -> b c h w')
        # vis = torch.sum(vis, dim=1)

        # vis = torch.sum(vis, dim=1, keepdim=False)
        # vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-9)
        # ret = vis.mean()
        # vis = vis.gt(ret)
        # vis = torch.where(vis > 0, 1, 0)

        pa = pixel_accuracy_batch(vis, mask)
        iou = iou_batch(vis, mask)
        if pa is None:
            continue

        pixel_acc.append(pa)
        intersection.append(iou)

        iterator.set_description('PixAcc: %.4f, IoU: %.4f' % (np.array(pixel_acc).mean() * 100,
                                                              np.array(intersection).mean() * 100))

    print('[Eval Summary]:')
    print('mPA: {:.2f}%, mIoU: {:.2f}%'.format(np.array(pixel_acc).mean() * 100,
                                               np.array(intersection).mean() * 100))


def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


if __name__ == "__main__":
    # imagenet_ds = VOCSegmentation('/Datasets/voc/', download=False)
    # imagenet_ds = ImagenetSegDataset()
    imagenet_ds = COCOSegDataset()

    loader = torch.utils.data.DataLoader(
        imagenet_ds,
        collate_fn=my_collate_fn,
        batch_size=1,  # must be 1 when ViT is used since Chefer's ViT codes doesn't support batch interpretation
        shuffle=False)

    model = resnet(pretrained=True).cuda()
    # model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, weights='COCO_WITH_VOC_LABELS_V1').cuda()
    # model = vgg19(pretrained=Tr ue).cuda()
    # model = vit_lrp(pretrained=True).cuda()
    # model = vit_lrp_exp(pretrained=True).cuda()
    # model = vgg19(pretrained=True).cuda()
    model.eval()

    eval(model, loader)
