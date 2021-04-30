import numpy as np
import torch
from itertools import product
import tqdm
import torch.autograd as autograd
from torch.nn import functional as F
from pixelcnnpp.layers import to_one_hot


def sample_from_discretized_mix_logistic_inverse_CDF(x, model, nr_mix, noise=[], u=None, clamp=True, bisection_iter=15, T=1):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    #added
    if len(noise) != 0:
        noise = noise.permute(0, 2, 3, 1)
    #added

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] / T
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * 3)
        u.uniform_(1e-5, 1. - 1e-5)
        u = torch.log(u) - torch.log(1. - u)

    u_r, u_g, u_b = torch.chunk(u, chunks=3, dim=-1)

    u_r = u_r.reshape(ls[:-1])
    u_g = u_g.reshape(ls[:-1])
    u_b = u_b.reshape(ls[:-1])

    log_softmax = torch.log_softmax(logit_probs, dim=-1)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix: 3 * nr_mix])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.) + np.log(T)
    if clamp:
        ubs = l.new_ones(ls[:-1])
        lbs = -ubs
    else:
        ubs = l.new_ones(ls[:-1]) * 20.
        lbs = -ubs

    means_r = means[..., 0, :]
    log_scales_r = log_scales[..., 0, :]

    def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_r) / log_scales_r.exp()

        if mode == 'cdf':
            log_logistic_cdf = -F.softplus(-centered_values)
            log_logistic_sf = -F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = -centered_values - log_scales_r - 2. * F.softplus(-centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x0 = binary_search(u_r, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_r(x, mode='cdf'), bisection_iter)

    if len(noise) == 0:
        means_g = x0.unsqueeze(-1) * coeffs[:, :, :, 0, :] + means[..., 1, :]
    else:
        means_g = (x0.unsqueeze(-1) + noise[:, :, :, 0].unsqueeze(-1)) * coeffs[:, :, :, 0, :] + means[..., 1, :]

    means_g = means_g.detach() #added, to make autograd sample correct
    log_scales_g = log_scales[..., 1, :]

    log_p_r, log_p_r_mixtures = log_cdf_pdf_r(x0, mode='pdf', mixtures=True)

    def log_cdf_pdf_g(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_g) / log_scales_g.exp()

        if mode == 'cdf':
            log_logistic_cdf = log_p_r_mixtures - log_p_r[..., None] - F.softplus(-centered_values)
            log_logistic_sf = log_p_r_mixtures - log_p_r[..., None] - F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = log_p_r_mixtures - log_p_r[..., None] - centered_values - log_scales_g - 2. * F.softplus(
                -centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x1 = binary_search(u_g, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_g(x, mode='cdf'), bisection_iter)

    if len(noise) == 0:
        means_b = x1.unsqueeze(-1) * coeffs[:, :, :, 2, :] + x0.unsqueeze(-1) * coeffs[:, :, :, 1, :] + means[..., 2, :]
    else:
        means_b = (x1.unsqueeze(-1) + noise[:, :, :, 1].unsqueeze(-1)) * coeffs[:, :, :, 2, :] + \
                  (x0.unsqueeze(-1) + noise[:, :, :, 0].unsqueeze(-1)) * coeffs[:, :, :, 1, :] + means[..., 2, :]

    means_b = means_b.detach() #added, to make autograd sample correct
    log_scales_b = log_scales[..., 2, :]

    log_p_g, log_p_g_mixtures = log_cdf_pdf_g(x1, mode='pdf', mixtures=True)

    def log_cdf_pdf_b(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_b) / log_scales_b.exp()

        if mode == 'cdf':
            log_logistic_cdf = log_p_g_mixtures - log_p_g[..., None] - F.softplus(-centered_values)
            log_logistic_sf = log_p_g_mixtures - log_p_g[..., None] - F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = log_p_g_mixtures - log_p_g[..., None] - centered_values - log_scales_b - 2. * F.softplus(
                -centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x2 = binary_search(u_b, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_b(x, mode='cdf'), bisection_iter)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def sample_from_discretized_mix_logistic_inverse_CDF_1d(x, model, nr_mix, u=None, clamp=True, bisection_iter=15):
    # Pytorch ordering

    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        l = model(x)
    # print(prof.key_averages().table(sort_by='cpu_time_total'))
    # breakpoint()

    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])
    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2])
        u.uniform_(1e-5, 1. - 1e-5)
        u = torch.log(u) - torch.log(1. - u)

    u_r = u.reshape(ls[:-1])

    log_softmax = torch.log_softmax(logit_probs, dim=-1)
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    if clamp is True:
        ubs = l.new_ones(ls[:-1]) * 1.
        lbs = -ubs
    else:
        ubs = l.new_ones(ls[:-1]) * 30.
        lbs = -ubs

    means_r = means[..., 0, :]
    log_scales_r = log_scales[..., 0, :]

    def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_r) / log_scales_r.exp()

        if mode == 'cdf':
            log_logistic_cdf = -F.softplus(-centered_values)
            log_logistic_sf = -F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = -centered_values - log_scales_r - 2. * F.softplus(-centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x0 = binary_search(u_r, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_r(x, mode='cdf'), bisection_iter)

    out = x0.view(xs[:-1] + [1])
    out = out.permute(0, 3, 1, 2)
    return out


def sample_from_discretized_mix_logistic_1d(x, model, nr_mix, u=None, clamp=True):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 1))
        u.uniform_(1e-5, 1. - 1e-5)
    mixture_u, sample_u = torch.split(u, [l.shape[1] * l.shape[2] * nr_mix,
                                          l.shape[1] * l.shape[2] * 1], dim=-1)
    mixture_u = mixture_u.reshape(l.shape[0], l.shape[1], l.shape[2], nr_mix)
    sample_u = sample_u.reshape(l.shape[0], l.shape[1], l.shape[2], 1)

    mixture_u = logit_probs.data - torch.log(- torch.log(mixture_u))
    _, argmax = mixture_u.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)

    x = means + torch.exp(log_scales) * (torch.log(sample_u) - torch.log(1. - sample_u))
    if clamp:
        x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    else:
        x0 = x[:, :, :, 0]

    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(x, model, nr_mix, u=None, T=1, clamp=True):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] / T
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 3))
        u.uniform_(1e-5, 1. - 1e-5)

    mixture_u, sample_u = torch.split(u, [l.shape[1] * l.shape[2] * nr_mix,
                                          l.shape[1] * l.shape[2] * 3], dim=-1)
    mixture_u = mixture_u.reshape(l.shape[0], l.shape[1], l.shape[2], nr_mix)
    sample_u = sample_u.reshape(l.shape[0], l.shape[1], l.shape[2], 3)

    mixture_u = logit_probs.data - torch.log(- torch.log(mixture_u))
    _, argmax = mixture_u.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.) + np.log(T)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x = means + torch.exp(log_scales) * (torch.log(sample_u) - torch.log(1. - sample_u))
    if clamp:
        x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    else:
        x0 = x[:, :, :, 0]

    if clamp:
        x1 = torch.clamp(torch.clamp(
            x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    else:
        x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0

    if clamp:
        x2 = torch.clamp(torch.clamp(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)
    else:
        x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def binary_search(log_cdf, lb, ub, cdf_fun, n_iter=15):
    with torch.no_grad():
        for i in range(n_iter):
            mid = (lb + ub) / 2.
            mid_cdf_value = cdf_fun(mid)
            right_idxes = mid_cdf_value < log_cdf
            left_idxes = ~right_idxes
            lb[right_idxes] = torch.min(mid[right_idxes], ub[right_idxes])
            ub[left_idxes] = torch.max(mid[left_idxes], lb[left_idxes])

    return mid