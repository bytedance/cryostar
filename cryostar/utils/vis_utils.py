import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt


def plt_twin(x,
             y1,
             y2,
             x_label,
             y1_label,
             y2_label,
             y1_lim=None,
             y2_lim=None,
             y1_color='tab:red',
             y2_color='tab:blue',
             save_path=None):
    # assume x is continuous, fill empty y with np.nan
    plt.clf()
    total_num = max(x) - min(x) + 1
    new_x = np.arange(min(x), max(x) + 1)
    new_y1 = np.empty(total_num, dtype=y1.dtype)
    new_y1.fill(np.nan)
    new_y2 = np.empty(total_num, dtype=y2.dtype)
    new_y2.fill(np.nan)

    new_y1[x - min(x)] = y1
    new_y2[x - min(x)] = y2

    fig, ax = plt.subplots()
    ax.plot(new_x, new_y1, color=y1_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y1_label)
    if y1_lim is not None:
        ax.set_ylim(y1_lim[0], y1_lim[1])
    ax.yaxis.label.set_color(y1_color)
    ax.tick_params(axis='y', colors=y1_color)

    ax2 = ax.twinx()
    ax2.plot(new_x, new_y2, color=y2_color)
    ax2.set_ylabel(y2_label)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim[0], y2_lim[1])
    ax.yaxis.label.set_color(y2_color)
    ax.tick_params(axis='y', colors=y2_color)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plt_single(x, y, x_label, y_label, y_color='tab:blue', x_lim=None, y_lim=None, save_path=None):
    # assume x is continuous, fill empty y with np.nan
    total_num = max(x) - min(x) + 1
    new_x = np.arange(min(x), max(x) + 1)
    new_y = np.empty(total_num, dtype=y.dtype)
    new_y.fill(np.nan)

    new_y[x - min(x)] = y

    fig, ax = plt.subplots()
    ax.plot(new_x, new_y, color=y_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # plot max and min value
    id_arr = np.argsort(y)
    ax.plot(x[id_arr[0]], y[id_arr[0]], '.', color='tab:orange')
    ax.text(x[id_arr[0]], y[id_arr[0]], f"{y[id_arr[0]]:.4f}")

    ax.plot(x[id_arr[-1]], y[id_arr[-1]], '.', color='tab:orange')
    ax.text(x[id_arr[-1]], y[id_arr[-1]], f"{y[id_arr[-1]]:.4f}")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plt_z(x, y, x_lim=None, y_lim=None, save_path=None):
    # plot z scatter
    plt.clf()
    plt.scatter(x, y, marker='.', s=2)

    u = np.unique(x)
    median_val_list = []
    for i in u:
        median_val_list.append(np.median(y[x == i]))

    plt.scatter(u, median_val_list, marker='.', color='tab:orange')
    plt.xlabel("true reaction coordinate")
    plt.ylabel("latent encoding")
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_z_dist(z, extra_cluster=None, save_path=None):
    if z.shape[-1] == 1:
        fig = sns.displot(x=z[:, 0])
        fig.set_xlabels("z values")
        if save_path is not None:
            fig.savefig(save_path)
    elif z.shape[-1] == 2:
        sns.set()
        fig = sns.jointplot(x=z[:, 0], y=z[:, 1], kind="kde", fill=True)
        ax = fig.figure.axes
        if extra_cluster is not None:
            ax[0].scatter(extra_cluster[:, 0], extra_cluster[:, 1], marker='.', color='tab:orange')
        if save_path is not None:
            fig.savefig(save_path)
    else:
        raise ValueError(f"input z with shape {z.shape}")


def save_tensor_image(tensors, save_path, mask=None):
    # normalize
    max_val = torch.max(tensors.flatten(start_dim=1), 1)[0][:, None, None, None]
    min_val = torch.min(tensors.flatten(start_dim=1), 1)[0][:, None, None, None]
    tensors = (tensors - min_val) / (max_val - min_val)

    show_img = ToPILImage()(make_grid(tensors, nrow=5))
    if mask is None:
        show_img.save(save_path)
    else:
        show_img = np.copy(np.asarray(show_img))
        # show_img = cv2.cvtColor(show_img, cv2.COLOR_GRAY2RGB)
        if mask.ndim == 2:
            mask = mask[None]
        mask = ToPILImage()(make_grid(mask.expand(tensors.shape[0], -1, -1, -1), nrow=5))
        mask = np.invert(np.asarray(mask).astype(bool))[..., 0]
        color_mask = np.array([[0, 0, 0], [31, 119, 180]], dtype=np.uint8)
        color_mask = color_mask[mask.astype(int)]
        show_img[mask] = cv2.addWeighted(show_img[mask], 0.5, color_mask[mask], 0.5, 0)
        show_img = Image.fromarray(show_img)
        show_img.save(save_path)