import torch
import torch.nn.functional as F
import numpy as np
import math


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        self.proxy_model = proxy_model
        self.lr_p = lr_proxy_model
        self.lr_w = lr_weights
        self.optimizer_theta_p_model = torch.optim.SGD(
            self.proxy_model.parameters(), lr=self.lr_p)
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5
        self.beta = beta
        self.buffer = []
        self.identity = []
        self.coef_inner = 5.0
        self.coef_intermediate = 5.0
        self.coef_outer = 5.0
        self.eps = 1e-8
        self.solver = 's-tfbo'

    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def train_inner(self, data_S, target_S, task_id, sample_weights,
                    inner_epchos):

        loss = math.inf
        if self.solver == 'original':
            for _ in range(inner_epchos):
                self.optimizer_theta_p_model = torch.optim.SGD(
                    self.proxy_model.parameters(), lr=self.lr_p)
                self.proxy_model.train()
                self.optimizer_theta_p_model.zero_grad()
                data = data_S.to(self.device).type(torch.float)
                target = target_S.to(self.device).type(torch.long)
                sample_weights = sample_weights.to(self.device).type(
                    torch.float).detach()
                output = self.proxy_model(data, task_id)
                loss = torch.mean(
                    sample_weights *
                    F.cross_entropy(output, target, reduction='none'))
                loss.backward()
                self.optimizer_theta_p_model.step()
                self.proxy_model.zero_grad()

        elif self.solver == 's-tfbo':
            self.proxy_model.train()
            self.proxy_model.zero_grad()
            data = data_S.to(self.device).type(torch.float)
            target = target_S.to(self.device).type(torch.long)
            sample_weights = sample_weights.to(self.device).type(
                torch.float).detach()
            output = self.proxy_model(data, task_id)
            loss = torch.mean(
                sample_weights *
                F.cross_entropy(output, target, reduction='none'))
            loss.backward()
            grad_inner = [
                p.grad.data.clone() for p in self.proxy_model.parameters()
            ]
            grad_inner_norm = torch.cat([g.flatten()
                                         for g in grad_inner]).norm().item()
            self.coef_inner = math.sqrt(self.coef_inner**2 +
                                        grad_inner_norm**2)
            with torch.no_grad():
                for p in self.proxy_model.parameters():
                    p.data -= (
                        1 /
                        (self.coef_inner + self.eps)) * p.grad.data.clone()
            self.proxy_model.zero_grad()

        elif self.solver == 'd-tfbo':
            for _ in range(inner_epchos):
                self.proxy_model.train()
                self.proxy_model.zero_grad()
                data = data_S.to(self.device).type(torch.float)
                target = target_S.to(self.device).type(torch.long)
                sample_weights = sample_weights.to(self.device).type(
                    torch.float).detach()
                output = self.proxy_model(data, task_id)
                loss = torch.mean(
                    sample_weights *
                    F.cross_entropy(output, target, reduction='none'))
                loss.backward()
                grad_inner = [
                    p.grad.data.clone() for p in self.proxy_model.parameters()
                ]
                grad_inner_norm = torch.cat([g.flatten() for g in grad_inner
                                             ]).norm().item()
                self.coef_inner = math.sqrt(self.coef_inner**2 +
                                            grad_inner_norm**2)
                with torch.no_grad():
                    for p in self.proxy_model.parameters():
                        p.data -= (1 / (self.coef_inner + self.eps)
                                   ) * p.grad.data.clone()
                self.proxy_model.zero_grad()
        else:
            raise ValueError(f'unsupported solver {self.solver}.')
        return loss

    def train_outer(self,
                    data,
                    target,
                    task_id,
                    data_weights,
                    topk,
                    ref_x=None,
                    ref_y=None):
        data = data.to(self.device)
        target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)
        X_S = data[:].to(self.device)
        y_S = target[:].to(self.device).type(torch.long)
        return self.update_sample_weights(data,
                                          target,
                                          task_id,
                                          X_S,
                                          y_S,
                                          sample_weights,
                                          topk,
                                          beta=self.beta,
                                          ref_x=ref_x,
                                          ref_y=ref_y)

    def update_sample_weights(self,
                              input_train,
                              target_train,
                              task_id,
                              input_selected,
                              target_selected,
                              sample_weights,
                              topk,
                              beta,
                              epsilon=1e-3,
                              ref_x=None,
                              ref_y=None):
        z = torch.normal(0, 1, size=[topk]).cuda()
        loss_outer = F.cross_entropy(self.proxy_model(input_train, task_id),
                                     target_train,
                                     reduction='none')
        topk_weights, ind = sample_weights.topk(topk)
        loss_outer_avg = torch.mean(loss_outer) - beta * (topk_weights +
                                                          epsilon * z).sum()
        if ref_x != None:
            loss_buff = []
            for i in range(task_id - 1):
                loss_buff += F.cross_entropy(self.proxy_model(
                    ref_x[i].to(self.device), i + 1),
                                             ref_y[i].to(self.device),
                                             reduction='none')
            loss_buff_avg = torch.mean(torch.Tensor(loss_buff))
            alpha = 0.1
            loss_outer_avg += alpha * loss_buff_avg
        d_theta = torch.autograd.grad(loss_outer_avg,
                                      self.proxy_model.parameters())
        v_0 = d_theta
        loss_inner = torch.mean(
            F.softmax(sample_weights, dim=-1) *
            F.cross_entropy(self.proxy_model(input_selected, task_id),
                            target_selected,
                            reduction='none'))
        grads_theta = torch.autograd.grad(loss_inner,
                                          self.proxy_model.parameters(),
                                          create_graph=True)
        G_theta = []
        for p, g in zip(self.proxy_model.parameters(), grads_theta):
            if g == None:
                G_theta.append(None)
            else:
                G_theta.append(p - self.lr_p * g)
        v_Q = v_0

        if self.sovler == 'original':
            for _ in range(3):
                v_new = torch.autograd.grad(G_theta,
                                            self.proxy_model.parameters(),
                                            grad_outputs=v_0,
                                            retain_graph=True)
                v_0 = [i.detach() for i in v_new]
                for i in range(len(v_0)):
                    v_Q[i].add_(v_0[i].detach())

            jacobian = -torch.autograd.grad(
                grads_theta, sample_weights, grad_outputs=v_Q)[0]
            with torch.no_grad():
                sample_weights -= self.lr_w * jacobian

        elif self.solver == 's-tfbo':
            v_new = torch.autograd.grad(G_theta,
                                        self.proxy_model.parameters(),
                                        grad_outputs=v_0,
                                        retain_graph=True)
            v_0 = [i.detach() for i in v_new]
            grad_intermediate_norm = torch.cat([g.flatten()
                                                for g in v_0]).norm().item()
            self.coef_intermediate = math.sqrt(self.coef_intermediate**2 +
                                               grad_intermediate_norm**2)
            phi = max(self.coef_inner, self.coef_intermediate)
            for i in range(len(v_0)):
                v_Q[i].sub_((1 / (phi + self.eps)) * v_0[i].detach())

            jacobian = -torch.autograd.grad(
                grads_theta, sample_weights, grad_outputs=v_Q)[0]
            grad_outer_norm = jacobian.norm().item()
            self.coef_outer = math.sqrt(self.coef_outer**2 +
                                        grad_outer_norm**2)
            with torch.no_grad():
                sample_weights -= 1 / (self.coef_outer * phi +
                                       self.eps) * jacobian

        elif self.solver == 'd-tfbo':
            for _ in range(3):
                v_new = torch.autograd.grad(G_theta,
                                            self.proxy_model.parameters(),
                                            grad_outputs=v_0,
                                            retain_graph=True)
                v_0 = [i.detach() for i in v_new]
                grad_intermediate_norm = torch.cat([g.flatten() for g in v_0
                                                    ]).norm().item()
                self.coef_intermediate = math.sqrt(self.coef_intermediate**2 +
                                                   grad_intermediate_norm**2)
                for i in range(len(v_0)):
                    v_Q[i].sub_((1 / (self.coef_intermediate + self.eps)) *
                                v_0[i].detach())

            jacobian = -torch.autograd.grad(
                grads_theta, sample_weights, grad_outputs=v_Q)[0]
            grad_outer_norm = jacobian.norm().item()
            self.coef_outer = math.sqrt(self.coef_outer**2 +
                                        grad_outer_norm**2)
            with torch.no_grad():
                sample_weights -= 1 / (self.coef_outer + self.eps) * jacobian

        else:
            raise ValueError(f'Unsupported solver {self.solver}.')

        return sample_weights, jacobian, loss_outer
