# -*- coding: utf-8 -*-
"""
 @time: 2024/1/25 15:48
 @desc:
"""
import torch
from tqdm import tqdm
import numpy as np
from CNNs.resnet import resnet18
from CNNs import dtd_opt as sa_opt
from data.imagenet import ImagenetDataset


def eval(model, loader):
    # net = sa_base.ActivationStoringNet(sa_base.model_flattening(model)).cuda()
    # DTD = sa_base.DTD().cuda()
    net = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    DTD = sa_opt.DTDOpt().cuda()

    num_samples = 0
    model_index = 0
    num_correct = np.zeros((len(imagenet_ds,)))
    num_correct_n = np.zeros((len(imagenet_ds, )))
    list_fx = np.zeros((len(imagenet_ds, )))
    list_fn = np.zeros((len(imagenet_ds, )))
    list_fs = np.zeros((len(imagenet_ds, )))

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        # Update the number of samples
        num_samples += len(data)
        target = target.to(data.device)
        # model inference
        module_stack, output = net(data.clone())  # [b, 1000]
        _ = DTD(module_stack, output, 1000, 'resnet')  # [b, 3, 224, 224]

        pred_class = output.data.max(1, keepdim=True)[1].squeeze(1)  # [b]
        tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()   # [b, ]
        num_correct[model_index:model_index+len(tgt_pred)] = tgt_pred
        fx = torch.gather(output, dim=1, index=target.unsqueeze(1))

        noise = DTD.get_noise()
        signal = data - noise
        _, output_s = net(signal)
        fs = torch.gather(output_s, dim=1, index=target.unsqueeze(1))
        _, output_n = net(noise)
        fn = torch.gather(output_n, dim=1, index=target.unsqueeze(1))

        fs_else = torch.gather(output_s, dim=1, index=torch.zeros_like(target.unsqueeze(1)))
        fx_else = torch.gather(output_s, dim=1, index=torch.zeros_like(target.unsqueeze(1)))

        list_fn[model_index:model_index + len(fn)] = fn.squeeze().data.cpu().numpy()
        list_fx[model_index:model_index + len(fx_else)] = fx_else.squeeze().data.cpu().numpy()
        list_fs[model_index:model_index + len(fs_else)] = fs_else.squeeze().data.cpu().numpy()

        print('f(x):     ', torch.mean(fx))
        print('f(noise): ', torch.mean(fn))

        pred_class_n = output_n.data.max(1, keepdim=True)[1].squeeze(1)  # [b]
        tgt_pred_n = (target == pred_class_n).type(target.type()).data.cpu().numpy()  # [b, ]
        num_correct_n[model_index:model_index + len(tgt_pred_n)] = tgt_pred_n

        model_index += len(target)

    print('Acc ', np.mean(num_correct))
    print('f(N) Acc ', np.mean(num_correct_n))
    with open('f_noise_acc.txt', 'w') as f:
        f.writelines(['%.5f ' % e+'\n' for e in list_fn])
    with open('f_signal_acc.txt', 'w') as f:
        f.writelines(['%.5f ' % e+'\n' for e in list_fs])
    with open('f_x_acc.txt', 'w') as f:
        f.writelines(['%.5f ' % e+'\n' for e in list_fx])
    with open('Batch.txt', 'w') as f:
        f.writelines(['%.5f ' % e + '\n' for e in range(len(list_fx))])


if __name__ == "__main__":
    imagenet_ds = ImagenetDataset()

    model = resnet18(pretrained=True).cuda()
    model.train(False)

    loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=16,
        shuffle=True)

    eval(model, loader)
