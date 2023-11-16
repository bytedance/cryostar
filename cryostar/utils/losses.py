import torch


def calc_cryoem_loss(pred_images, gt_images, mask=None, reduction="mean"):
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]
    delta = pred_images.flatten(start_dim=1) - gt_images.flatten(start_dim=1)
    loss = torch.sum(torch.pow(delta, 2), dim=1)  # (bsz, ) image-wise
    loss /= pixel_num  # (bsz, ) pixel-wise
    if reduction == "mean":
        return torch.mean(loss)  # averaged over bsz x pixel
    elif reduction == "none":
        return loss
    else:
        raise NotImplemented


def calc_cor_loss(pred_images, gt_images, mask=None):
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]

    # b, c, h, w -> b, c, num_pix
    pred_images = pred_images.flatten(start_dim=2)
    gt_images = gt_images.flatten(start_dim=2)

    # b, c
    dots = (pred_images * gt_images).sum(-1)
    # b, c -> b, c
    err = -dots / (gt_images.std(-1) + 1e-5) / (pred_images.std(-1) + 1e-5)
    # b, c -> b -> 1 value
    err = err.sum(-1).mean() / pixel_num
    return err


def calc_kl_loss(mu, log_var, free_bits, reduction="mean"):
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    # free bits
    kld_loss = torch.clamp(kld_loss, free_bits)  # (bsz, z-dim)
    kld_loss = torch.mean(kld_loss, dim=1)  # (bsz, )
    if reduction == "mean":
        kld_loss = torch.mean(kld_loss)  # averaged over bsz x z-dim
    elif reduction == "none":
        kld_loss = kld_loss
    else:
        raise NotImplementedError
    return kld_loss