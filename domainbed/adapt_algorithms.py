# The code is modified from domainbed.algorithms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

from domainbed.algorithms import Algorithm
from domainbed.sam import SAM

from argparse import Namespace
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
import math


ALGORITHMS = [
    # 'T3A', 
    'Tent', 
    # 'TentNorm',  
    # 'TentPreBN',  # Tent-BN in the paper
    # 'TentClf',  # Tent-C in the paper
    # 'PseudoLabel', 
    # 'PLClf', 
    # 'SHOT', 
    # 'SHOTIM',
    'SAR',
    'DeYO',
    'COME',
    'DSBR',
    'ROID',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class DeYO(Algorithm):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, lr=hparams['lr'])
        self.steps = hparams.get('num_update_steps', 1)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = False

        self.margin_e0 = hparams.get('margin_e0', 0.4) * math.log(num_classes)
        self.deyo_margin = hparams.get('deyo_margin', 0.5) * math.log(num_classes)
        
        self.args = Namespace()
        self.args.__dict__.update({
            'filter_ent': 1,
            'filter_plpd': 1,
            'plpd_threshold': 0.2,
            'reweight_ent': 1,
            'reweight_plpd': 1,
            'row_start': 56,
            'column_start': 56,
            'occlusion_size': 112,
            'patch_len': 4,
            'aug_type': 'patch'  # 'occ', 'patch', 'pixel', or None
        })
        self.args.__dict__.update(hparams)
        
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False, targets=None, group=None):
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            outputs = forward_and_adapt_deyo(x, self.model, self.args,
                                            self.optimizer, self.deyo_margin,
                                            self.margin_e0, targets, adapt, group)
        
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def configure_model_optimizer(self, algorithm, lr):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        return adapted_algorithm, optimizer


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo(x, model, args, optimizer, deyo_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0
        
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    # optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

# from https://github.com/mr-eggplant/SAR/blob/main/sar.py
def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema, entropy_fn):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = entropy_fn(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    entropys2 = entropy_fn(model(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < reset_constant:
            # print("ema < reset_constant, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag, len(filter_ids_1[0]), len(filter_ids_2[0])

class SAR(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.use_sam = hparams.get('use_sam', True)  # Control SAM usage via hparams
        self.use_come = hparams.get('use_come', False)  # Control COME entropy via hparams
        if self.use_come:
            self.entropy_fn = entropy_of_opinion
        else:
            self.entropy_fn = softmax_entropy
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, lr=hparams['lr'])
        self.steps = hparams.get('num_update_steps', 1)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = False

        self.margin_e0 = hparams.get('margin_e0', 0.4) * np.log(num_classes)
        self.reset_constant_em = hparams.get('reset_constant_em', 0.2)
        self.ema = None

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        num_filtered_1, num_filtered_2 = [], []
        
        if adapt:
            if self.episodic:
                self.reset()

            forward_fn = forward_and_adapt_sar if self.use_sam else forward_and_adapt_sar_no_sam
            for _ in range(self.steps):
                outputs, ema, reset_flag, filtered_1, filtered_2 = forward_fn(
                    x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema, self.entropy_fn
                )
                num_filtered_1.append(filtered_1)
                num_filtered_2.append(filtered_2)
                if reset_flag:
                    self.reset()
                self.ema = ema  # update moving average value of loss
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)

        return outputs, num_filtered_1, num_filtered_2

    def configure_model_optimizer(self, algorithm, lr):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        
        if self.use_sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar_no_sam(x, model, optimizer, margin, reset_constant, ema, entropy_fn):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = entropy_fn(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()
    optimizer.step()
    # optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    # entropys2 = softmax_entropy(model(x))
    # entropys2 = entropys2[filter_ids_1]  # second time forward  
    # loss_second_value = entropys2.clone().detach().mean(0)
    # filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    # loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss.item()):
        ema = update_ema(ema, loss.item())  # record moving average loss values for model recovery

    # # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    # loss_second.backward()
    # optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < reset_constant:
            # print("ema < reset_constant, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag, len(filter_ids_1[0]), 0



class Tent(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.base_optimizer = hparams.get('base_optimizer', 'SGD')
        self.use_sam = hparams.get('use_sam', False)  # Control SAM usage via hparams
        self.use_come = hparams.get('use_come', False)  # Control COME usage via hparams
        if self.use_come:
            self.entropy_fn = entropy_of_opinion
        else: 
            self.entropy_fn = softmax_entropy
            
        self.num_classes = num_classes
        
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, lr=hparams['lr'])
        self.steps = hparams.get('num_update_steps', 1)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            forward_fn = self.forward_and_adapt_sam if self.use_sam else self.forward_and_adapt
            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = forward_fn(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = forward_fn(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        entropys = self.entropy_fn(outputs)
        loss = entropys.mean(0)
        loss.backward()
        optimizer.step()
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sam(self, x, model, optimizer):
        """Forward and adapt model on batch of data using SAM."""
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        entropys = self.entropy_fn(outputs)
        loss = entropys.mean(0)
        loss.backward()
        optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        outputs_second = model(x)
        entropys_second = self.entropy_fn(outputs_second)
        loss_second = entropys_second.mean(0)
        loss_second.backward()  # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        optimizer.second_step(zero_grad=True)
        
        return outputs
    
    def configure_model_optimizer(self, algorithm, lr):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        
        if self.use_sam:
            assert self.base_optimizer in 'SGD', "currently only SGD is supported as the base optimizer for SAM in this implementation"
            base_optimizer = getattr(torch.optim, self.base_optimizer)
            optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
        else:
            if self.base_optimizer == 'Adam':
                optimizer = torch.optim.Adam(
                    params, 
                    lr=lr,
                )
            elif self.base_optimizer == 'SGD':
                optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state) 

class DSBR(Tent):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        hparams["base_optimizer"] = hparams.get('base_optimizer', 'Adam')
        hparams["use_sam"] = False
        super().__init__(input_shape, num_classes, num_domains, hparams, algorithm)  
        self.cluster_ditribution = torch.ones(num_classes) / num_classes
        self.ema_alpha = hparams.get('ema_alpha', 0.9)
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        entropies = self.entropy_fn(outputs)
        
        cluster_ids = outputs.argmax(dim=1)
        local_cluster_ditribution = torch.bincount(cluster_ids, minlength=self.num_classes) / len(cluster_ids)
        self.cluster_ditribution = self.ema_alpha * self.cluster_ditribution.to(entropies.device) + (1 - self.ema_alpha) * local_cluster_ditribution
        entropies = entropies / (self.num_classes * self.cluster_ditribution[cluster_ids])

        loss = entropies.mean(0)
        loss.backward()
        optimizer.step()
        return outputs
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.cluster_ditribution = torch.ones_like(self.cluster_ditribution) / len(self.cluster_ditribution)

def entropy_of_opinion_no_reg(x: torch.Tensor): #key component of COME
    assert len(x.shape) == 2
    K = x.shape[-1]
    
    evidence = torch.exp(x)
    strength = torch.sum(evidence+1, dim=1, keepdim=True)
    belief = evidence / strength
    uncertainty = K / strength
    
    opinion = torch.cat([belief, uncertainty], dim=1) + 1e-7
    entropy = -(opinion * torch.log(opinion)).sum(1)
    return entropy

def entropy_of_opinion(x: torch.Tensor): #key component of COME
    assert len(x.shape) == 2
    K = x.shape[-1]
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    
    evidence = torch.exp(x)
    strength = torch.sum(evidence+1, dim=1, keepdim=True)
    belief = evidence / strength
    uncertainty = K / strength
    
    opinion = torch.cat([belief, uncertainty], dim=1) + 1e-7
    entropy = -(opinion * torch.log(opinion)).sum(1)
    return entropy


class COME(Tent):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        self.use_reg = hparams.get('use_reg', True)  # Control regularization via hparams
        super().__init__(input_shape, num_classes, num_domains, hparams, algorithm)
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        loss_fn = entropy_of_opinion if self.use_reg else entropy_of_opinion_no_reg
        entropies = loss_fn(outputs)
        entropies, _ = self._select_entropies(entropies, outputs=outputs, update_stats=True)
        loss = entropies.mean(0)
        # loss = entropies.sum() / num_classes
        loss.backward()
        optimizer.step()
        return outputs


class Entropy(nn.Module):
    def forward(self, logits):
        return softmax_entropy(logits)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, x_ema):
        return -(1 - self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - \
               self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.clip = clip

    def forward(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return -(probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, update_all=False):
    if momentum < 1.0:
        for name, param_to_update in model_to_update.named_parameters():
            if param_to_update.requires_grad or update_all:
                src_param = model_to_merge[name].to(param_to_update.device)
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * src_param
    return model_to_update


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)


class ColorJitterPro(torchvision.transforms.ColorJitter):
    """ColorJitter extension adding gamma jitter."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    def forward(self, img):
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(self.brightness[0], self.brightness[1]).item()
                img = TF.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(self.contrast[0], self.contrast[1]).item()
                img = TF.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(self.saturation[0], self.saturation[1]).item()
                img = TF.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(self.hue[0], self.hue[1]).item()
                img = TF.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma_factor = torch.tensor(1.0).uniform_(self.gamma[0], self.gamma[1]).item()
                img = img.clamp(1e-8, 1.0)
                img = TF.adjust_gamma(img, gamma_factor)

        return img


def get_roid_tta_transforms(img_size, gaussian_std=0.005, soft=False, padding_mode='edge', cotta_augs=False):
    n_pixels = img_size[0] if isinstance(img_size, (list, tuple)) else img_size

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        torchvision.transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        torchvision.transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            fill=0
        )
    ]

    if cotta_augs:
        tta_transforms += [
            torchvision.transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
            torchvision.transforms.CenterCrop(size=n_pixels),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            GaussianNoise(0, gaussian_std),
            Clip(0.0, 1.0)
        ]
    else:
        tta_transforms += [
            torchvision.transforms.CenterCrop(size=n_pixels),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            Clip(0.0, 1.0)
        ]

    return torchvision.transforms.Compose(tta_transforms)

class ROID(Algorithm):
    """ROID adaptation with reliability weighting, source EMA anchoring, and prior correction."""

    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, lr=hparams['lr'])
        self.steps = hparams.get('num_update_steps', 1)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = False

        self.num_classes = num_classes
        self.use_weighting = hparams.get('use_weighting', True)
        self.use_prior_correction = hparams.get('use_prior_correction', True)
        self.use_consistency = hparams.get('use_consistency', True)
        self.momentum_src = hparams.get('momentum_src', 0.99)
        self.momentum_probs = hparams.get('momentum_probs', 0.9)
        self.temperature = hparams.get('temperature', 1/3)

        img_size = input_shape[-2:] if len(input_shape) >= 3 else (224, 224)
        self.tta_transform = get_roid_tta_transforms(
            img_size=img_size,
            padding_mode="reflect",
            cotta_augs=False,
        )

        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()

        self.register_buffer('class_probs_ema', torch.ones(num_classes) / num_classes)

        # Store source parameters on CPU to anchor adaptation without extra GPU copy.
        self.src_param_state = {
            name: p.detach().cpu().clone()
            for name, p in self.model.named_parameters()
        }

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def configure_model_optimizer(self, algorithm, lr):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, _ = self.collect_params(adapted_algorithm.featurizer)
        if len(params) == 0:
            raise RuntimeError('ROID found no trainable normalization parameters in the featurizer.')

        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        return adapted_algorithm, optimizer

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier)
                else:
                    outputs = self.forward_and_adapt(x, self.model)
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)

        return outputs

    def loss_calculation(self, x, model):
        outputs = model(x)
        batch_size = outputs.shape[0]

        weights = torch.ones(outputs.shape[0], device=outputs.device)
        mask = torch.zeros(outputs.shape[0], dtype=torch.bool, device=outputs.device)

        if self.use_weighting:
            with torch.no_grad():
                weights_div = 1 - F.cosine_similarity(
                    self.class_probs_ema.unsqueeze(0),
                    outputs.softmax(1),
                    dim=1,
                )
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                weights_cert = -self.softmax_entropy(outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.0

                updated_probs = update_model_probs(
                    x_ema=self.class_probs_ema,
                    x=outputs.softmax(1).mean(0),
                    momentum=self.momentum_probs,
                )
                self.class_probs_ema.copy_(updated_probs)

        loss_out = self.slr(outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / batch_size

        # Consistency is only defined for 2D RGB-like inputs where torchvision TTA is valid.
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(x[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / batch_size

        return outputs, loss

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model):
        self.optimizer.zero_grad()
        outputs, loss = self.loss_calculation(x, model)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_param_state,
            momentum=self.momentum_src,
        )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.class_probs_ema.fill_(1.0 / self.num_classes)
        
    def collect_params(self, model):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None  
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True) 
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names
