import torch


def merge_step_outputs(outputs):
    ks = outputs[0].keys()
    res = {}
    for k in ks:
        res[k] = torch.concat([out[k] for out in outputs], dim=0)
    return res


def squeeze_dict_outputs_1st_dim(outputs):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k].flatten(start_dim=0, end_dim=1)
    return res


def filter_outputs_by_indices(outputs, indices):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k][indices]
    return res


def get_1st_unique_indices(t):
    _, idx, counts = torch.unique(t, dim=None, sorted=True, return_inverse=True, return_counts=True)
    # ind_sorted: the index corresponding to same unique value will be grouped by these indices
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((cum_sum.new_tensor([
        0,
    ]), cum_sum[:-1]))
    first_idx = ind_sorted[cum_sum]
    return first_idx