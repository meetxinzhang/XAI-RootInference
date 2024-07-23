# encoding: utf-8
from PIL import Image
import matplotlib.pyplot as plt
import torch
import glob
import platform
from data.__init__ import name_dict, idx_dict
import random
import os
from torchvision import transforms
from models.vit import vit_base_patch16_224 as vit_lrp
from models.vit_exp import vit_base_patch16_224 as vit_lrp_exp
from models.vit_lrp_manager import ignite_relprop, visualize_attention

print(torch.__version__, torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
resize_func = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

normal_func = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def print_top_classes(predictions):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, name_dict[cls_idx])
        output_string += ' ' * (max_str_len - len(name_dict[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    return class_indices


def file_scanf2(path, contains, endswith, is_random=False, sub_ratio=1.0):
    files = glob.glob(path + '/*')
    if is_random:
        random.shuffle(files)
    input_files = []
    for f in files[:int(len(files) * sub_ratio)]:
        if platform.system().lower() == 'windows':
            f.replace('\\', '/')
        if not any([c in f.split('/')[-1] for c in contains]):
            continue
        if not f.endswith(endswith):
            continue
        input_files.append(f)
    return input_files


if __name__ == "__main__":
    data_path = 'data/samples'
    file_names = []
    # for root, dirs, files in os.walk(data_path):
    #     for sub_dir in files:
    #         imgs = file_scanf2(path=data_path + '/' + sub_dir,
    #                            contains=[
    #                                'n',
    #                                # 'n01443537_9651', 'n01443537_3032', 'n01443537_1219', 'n01443537_6620',
    #                                # 'n01443537_11018', 'n01443537_11018', 'n01443537_16422', 'n01443537_21526',
    #                                # 'n01537544_2375', 'n01582220_392', 'n01614925_2400', 'n02113712_10565',
    #                                # 'n02690373_4225'
    #                            ],
    #                            endswith='.JPEG',
    #                            is_random=True, sub_ratio=1)
    #         file_names.extend(imgs)
    imgs = file_scanf2(path=data_path,
                       contains=[
                           'n',
                       ],
                       endswith='.JPEG',
                       is_random=True, sub_ratio=1)
    file_names.extend(imgs)

    # model1 = resnet18(pretrained=True).cuda()
    # model1.train(False)
    model = vit_lrp(pretrained=True).cuda()
    model_exp = vit_lrp_exp(pretrained=True).cuda()

    # act_net_base = sa_base.ActivationStoringNet(sa_base.model_flattening(model1)).cuda()
    # DTD = sa_base.DTD().cuda()
    # act_net_opt = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model1)).cuda()
    # OptDTD = sa_opt.DTDOpt().cuda()

    for f in file_names:
        image = Image.open(f)
        if image.mode != 'RGB':
            continue
        re_img = resize_func(image)
        input_tensor = normal_func(re_img)

        input_tensor = torch.autograd.Variable(input_tensor.unsqueeze(0).cuda())

        output = model(input_tensor)
        _ = model_exp(input_tensor)
        class_indices = print_top_classes(output)
        pre_idx = class_indices[0]
        # pre_dir = dir_dict[pre_idx]
        true_dir = f.split('/')[-1].split('_')[0]
        true_idx = idx_dict[true_dir]
        if true_idx == pre_idx:
            title = 'Y'
        else:
            title = 'N'

        # cat - the predicted class
        base_cam = ignite_relprop(model, input=input_tensor,
                                  method="transformer_attribution", alpha=1,
                                  index=true_idx).detach()
        exp_cam = ignite_relprop(model_exp, input=input_tensor,
                                 method="transformer_attribution", alpha=1,
                                 index=true_idx).detach()
        vis1 = visualize_attention(input_tensor, base_cam)
        vis2 = visualize_attention(input_tensor, exp_cam)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(re_img)
        axs[0].axis('off')
        axs[0].set_title(title + '-' + name_dict[true_idx])
        axs[1].imshow(vis1)
        axs[1].axis('off')
        axs[1].set_title('Baseline')
        axs[2].imshow(vis2)
        axs[2].axis('off')
        axs[2].set_title('Our')

        # axs[2].set_title(CLS2IDX[class_indices[1]])

        save_name = f.split('/')[-1].replace('.JPEG', '_')
        plt.savefig('/Datasets/ImageNet/test_vit'
                    '/' + save_name + ".jpg")
        plt.clf()
        # plt.close('all')
