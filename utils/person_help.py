import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.init as init
from matplotlib.colors import LinearSegmentedColormap

# 底色为黄色，值越高越偏暗红
cdict = {
    'red': [(0.0, 1.0, 1.0),  # 从黄色的红色通道开始（1.0）
            (1.0, 0.5, 0.5)],  # 最终变为暗红色（0.5）

    'green': [(0.0, 1.0, 1.0),  # 从黄色的绿色通道开始（1.0）
              (1.0, 0.0, 0.0)],  # 最终变为红色的绿色通道（0.0）

    'blue': [(0.0, 0.0, 0.0),  # 黄色的蓝色通道为0
             (1.0, 0.0, 0.0)],  # 最终变为红色的蓝色通道（0.0）
}
custom_cmap = LinearSegmentedColormap('YellowToDarkRed', cdict)


def purevis(x):
    # color_map = 'gray_r'
    plt.imshow(x.detach().cpu().squeeze().numpy())
    plt.xticks([])
    plt.yticks([])
    plt.show()


def c3vis(x):
    # x is a iamge with 3 channels, and the shape is [3, H, W]
    x = x.detach().cpu().numpy()
    plt.imshow(x.transpose(1, 2, 0))
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


def savefeatvis(fmap, save):
    fmap = fmap.detach().squeeze()
    plt.imshow(torch.norm(fmap, dim=0).cpu().numpy())
    plt.savefig(save)


def vis(img_list, name_list, ticks=False, colorbar=False, cmap='viridis', save_dir=None):
    assert len(img_list) == len(name_list)
    # 将img_list中的图片同时显示
    fig, ax = plt.subplots(1, len(img_list), figsize=(5 * len(img_list), 5))
    vmax = max([torch.max(img).item() for img in img_list])
    vmin = min([torch.min(img).item() for img in img_list])
    for i in range(len(img_list)):
        im = img_list[i]
        if isinstance(im, torch.Tensor):
            im = im.detach().squeeze().cpu().numpy()
        im_plot = ax[i].imshow(im, cmap=cmap, vmax=vmax, vmin=vmin)
        if ticks:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        if colorbar and i == len(img_list) - 1:
            fig.colorbar(im_plot, ax=ax[i])
        if name_list is not None:
            ax[i].set_title(name_list[i])
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close()


def Initialize_net(net, mode='equal'):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            if mode == 'equal':
                init.constant_(layer.weight, 1 / (layer.out_channels))
            else:
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0.1)


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)
    img_list = [x, x]
    save_dir = '1.png'
    vis(img_list, name_list=['gt', 'pred'], save_dir=save_dir, colorbar=True)
    pass
