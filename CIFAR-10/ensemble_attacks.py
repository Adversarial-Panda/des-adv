import torch
import torch.nn.functional as F
from typing import List

def ensemble_mi_fgsm(
    models: List[torch.nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8/255,
    alpha: float = 2/255,
    iters: int = 10,
    decay: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
    device: str = None,
):
    if device is None:
        device = x.device

    if not isinstance(models, (list, tuple)):
        models = [models]

    for m in models:
        m.to(device).eval()

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    x_orig = x.clone().detach().to(device).float()
    x_adv = x_orig.clone().detach()
    momentum = torch.zeros_like(x_adv).to(device)
    y = y.to(device)

    for _ in range(iters):
        x_adv.requires_grad_(True)

        # ----- Liu et al. (2017): sum/average logits before loss -----
        sum_logits = None
        for m in models:
            out = m(x_adv)
            if isinstance(out, (tuple, list)):
                out = out[0]
            sum_logits = out if sum_logits is None else sum_logits + out
        avg_logits = sum_logits / len(models)
        total_loss = loss_fn(avg_logits, y)
        # --------------------------------------------------------------

        grad = torch.autograd.grad(total_loss, x_adv, retain_graph=False, create_graph=False)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-8)

        # momentum update (MI-FGSM)
        momentum = decay * momentum + grad
        step = alpha * momentum.sign()

        x_adv = x_adv.detach() + step.detach()
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, min=clip_min, max=clip_max).detach()

    return x_adv




import torch
import torch.nn.functional as F
from typing import List
import random

def ensemble_svre_mi_fgsm(
    models: List[torch.nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8/255,
    alpha: float = 2/255,
    iters: int = 10,
    decay: float = 1.0,
    sample_k: int = 2,        # how many models to sample for stochastic update
    refresh: int = 5,         # snapshot frequency (every `refresh` iterations recompute full grad)
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
    device: str = None,
):
    """
    SVRE-like ensemble MI-FGSM (SVRG-based variance reduced estimator).
    - models: list of models (logit-ensemble formulation)
    - sample_k: number of models sampled per inner update (k <= len(models))
    - refresh: every `refresh` iterations compute full gradient snapshot g_tilde at x_tilde
    """
    if device is None:
        device = x.device

    if not isinstance(models, (list, tuple)):
        models = [models]

    n_models = len(models)
    for m in models:
        m.to(device).eval()

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    x_orig = x.clone().detach().to(device).float()
    x_adv = x_orig.clone().detach()
    momentum = torch.zeros_like(x_adv).to(device)
    y = y.to(device)

    # helper: compute gradient of avg-logits loss for a given subset of model indices at given inputs
    def grad_of_models_at_x(x_in, model_indices):
        x_in = x_in.clone().detach().requires_grad_(True)
        sum_logits = None
        for idx in model_indices:
            out = models[idx](x_in)
            if isinstance(out, (tuple, list)):
                out = out[0]
            sum_logits = out if sum_logits is None else sum_logits + out
        avg_logits = sum_logits / len(model_indices)
        loss = loss_fn(avg_logits, y)
        g = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False)[0]
        return g.detach()

    # initial full gradient snapshot g_tilde at x_adv
    x_tilde = x_adv.clone().detach()
    g_tilde = grad_of_models_at_x(x_tilde, list(range(n_models)))

    for t in range(iters):
        # refresh snapshot occasionally
        if t % refresh == 0 and t != 0:
            x_tilde = x_adv.clone().detach()
            g_tilde = grad_of_models_at_x(x_tilde, list(range(n_models)))

        # sample subset of models (without replacement)
        if sample_k >= n_models:
            sample_idx = list(range(n_models))
        else:
            sample_idx = random.sample(range(n_models), sample_k)

        # compute grad on sampled models at current x_adv
        g_sample = grad_of_models_at_x(x_adv, sample_idx)
        # compute grad on sampled models at snapshot x_tilde
        g_sample_tilde = grad_of_models_at_x(x_tilde, sample_idx)

        # SVRG estimator
        g_hat = g_tilde + (g_sample - g_sample_tilde)
        # normalize (L1) and momentum update (MI)
        g_hat = g_hat / (torch.norm(g_hat, p=1) + 1e-8)
        momentum = decay * momentum + g_hat

        step = alpha * momentum.sign()
        x_adv = x_adv.detach() + step.detach()
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, min=clip_min, max=clip_max).detach()

    return x_adv



from abc import abstractmethod

import torch
import torch.nn.functional as F


class AdaEA_Base:
    def __init__(self, models, eps=8/255, alpha=2/255, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        assert isinstance(models, list) and len(models) >= 2, 'Error'
        self.device = device
        self.models = models
        self.num_models = len(self.models)
        for model in models:
            model.eval()

        # attack parameter
        self.eps = eps
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.beta = beta
        self.alpha = alpha

    def get_adv_example(self, ori_data, adv_data, grad, attack_step=None):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        if attack_step is None:
            adv_example = adv_data.detach() + grad.sign() * self.alpha
        else:
            adv_example = adv_data.detach() + grad.sign() * attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    def agm(self, ori_data, cur_adv, grad, label):
        """
        Adaptive gradient modulation
        :param ori_data: natural images
        :param cur_adv: adv examples in last iteration
        :param grad: gradient in this iteration
        :param label: ground truth
        :return: coefficient of each model
        """
        loss_func = torch.nn.CrossEntropyLoss()

        # generate adversarial example
        adv_exp = [self.get_adv_example(ori_data=ori_data, adv_data=cur_adv, grad=grad[idx])
                   for idx in range(self.num_models)]
        loss_self = [loss_func(self.models[idx](adv_exp[idx]), label) for idx in range(self.num_models)]
        w = torch.zeros(size=(self.num_models,), device=self.device)

        for j in range(self.num_models):
            for i in range(self.num_models):
                if i == j:
                    continue
                w[j] += loss_func(self.models[i](adv_exp[j]), label) / loss_self[i] * self.beta
        w = torch.softmax(w, dim=0)

        return w

    def drf(self, grads, data_size):
        """
        disparity-reduced filter
        :param grads: gradients of each model
        :param data_size: size of input images
        :return: reduce map
        """
        reduce_map = torch.zeros(size=(self.num_models, self.num_models, data_size[0], data_size[-2], data_size[-1]),
                                 dtype=torch.float, device=self.device)
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        reduce_map_result = torch.zeros(size=(self.num_models, data_size[0], data_size[-2], data_size[-1]),
                                        dtype=torch.float, device=self.device)
        for i in range(self.num_models):
            for j in range(self.num_models):
                if i >= j:
                    continue
                reduce_map[i][j] = sim_func(F.normalize(grads[i], dim=1), F.normalize(grads[j], dim=1))
            if i < j:
                one_reduce_map = (reduce_map[i, :].sum(dim=0) + reduce_map[:, i].sum(dim=0)) / (self.num_models - 1)
                reduce_map_result[i] = one_reduce_map

        return reduce_map_result.mean(dim=0).view(data_size[0], 1, data_size[-2], data_size[-1])

    @abstractmethod
    def attack(self,
               data: torch.Tensor,
               label: torch.Tensor,
               idx: int = -1) -> torch.Tensor:
        ...

    def __call__(self, data, label, idx=-1):
        return self.attack(data, label, idx)





import torch
import torch.nn as nn


class AdaEA_MIFGSM(AdaEA_Base):
    def __init__(self, models, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), momentum=0.9):
        super().__init__(models=models, eps=eps, alpha=alpha, max_value=max_value, min_value=min_value,
                         threshold=threshold, device=device, beta=beta)
        self.iters = iters
        self.momentum = momentum

    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        grad_mom = torch.zeros_like(data, device=self.device)

        for i in range(self.iters):
            adv_data.requires_grad = True

            outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
            losses = [loss_func(outputs[idx], label) for idx in range(len(self.models))]
            grads = [torch.autograd.grad(losses[idx], adv_data, retain_graph=True, create_graph=False)[0]
                     for idx in range(len(self.models))]

            # AGM
            alpha = self.agm(ori_data=data, cur_adv=adv_data, grad=grads, label=label)

            # DRF
            cos_res = self.drf(grads, data_size=(B, C, H, W))
            cos_res[cos_res >= self.threshold] = 1.
            cos_res[cos_res < self.threshold] = 0.

            output = torch.stack(outputs, dim=0) * alpha.view(self.num_models, 1, 1)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]
            grad = grad * cos_res

            # momentum
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + self.momentum * grad_mom
            grad_mom = grad

            # add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data


import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

class Weight_Selection(nn.Module):
    def __init__(self, weight_len) -> None:
        super(Weight_Selection,self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones([weight_len]))

    def forward(self, x, index):
        x = self.weight[index] * x
        return x

def MI_FGSM_SMER(surrogate_models,images, labels, args,num_iter = 10):
    eps = args.eps/255.0
    alpha = args.alpha/255.0
    beta = alpha
    momentum = args.momentum
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    m = len(surrogate_models) 
    m_smer = m*4
    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(),lr=2e-2,weight_decay=2e-3)
    noise = 0
    grad = 0
    for i in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        images = Variable(images, requires_grad = True)
        x_inner = images.detach()
        x_before = images.clone()
        noise_inner_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        grad_inner = torch.zeros_like(images)
        options = []
        for i in range(int(m_smer / m)):
            options_single=[j for j in range(m)]
            np.random.shuffle(options_single)
            options.append(options_single)
        options = np.reshape(options,-1)
        for j in range(m_smer):
            option = options[j]
            grad_single = surrogate_models[option]
            x_inner.requires_grad = True
            out_logits = grad_single(x_inner)
            if type(out_logits) is list:
                out = weight_selection(out_logits[0],option)
                aux_out = weight_selection(out_logits[1],option)
            else:
                out = weight_selection(out_logits,option)
            loss = F.cross_entropy(out, labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(aux_out, labels)
            noise_im_inner = torch.autograd.grad(loss,x_inner)[0]
            group_logits = 0
            group_aux_logits = 0
            for m_step, model_s in enumerate(surrogate_models):
                out_logits = model_s(x_inner)
                if type(out_logits) is list:
                    logits = weight_selection(out_logits[0],m_step)
                    aux_logits = weight_selection(out_logits[1],m_step)
                else:
                    logits = weight_selection(out_logits,m_step)
                group_logits = group_logits + logits / m
                if type(out_logits) is list:
                    group_aux_logits = group_aux_logits + aux_logits / m
            loss = F.cross_entropy(group_logits,labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(group_aux_logits,labels)
            outer_loss = -torch.log(loss)
            x_inner.requires_grad = False
            outer_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            noise_inner = noise_im_inner
            noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), dim=[1, 2, 3], keepdims=True)
            grad_inner = grad_inner + noise_inner
            x_inner = x_inner + beta * torch.sign(grad_inner)
            x_inner = clip_by_tensor(x_inner, image_min, image_max)
            noise_inner_all[j] = grad_inner.clone()
        noise =noise_inner_all[-1].clone() 
        noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdims=True)
        grad = noise + momentum * grad
        images = x_before +  alpha * torch.sign(grad)
        images = clip_by_tensor(images, image_min, image_max)
    return images

