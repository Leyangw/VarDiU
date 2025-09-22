import torch
from torch import nn
import normflows as nf

class NeuralSplineFlow(nn.Module):
    def __init__(self, dim=2, flow_length=16, hidden_layers=2, hidden_units=128):
        super().__init__()
        self.dim = dim

        # t-conditioning from sigma
        self.net1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, dim)  # translation vector
        )

        # combine (z0, x) -> feature in R^dim
        self.net2 = nn.Sequential(
            nn.Linear(2 * dim, 32),  # expects concat of z0 and x
            nn.SiLU(),
            nn.Linear(32, dim)
        )

        flows = []
        for _ in range(flow_length):
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(dim, hidden_layers, hidden_units ))
            flows.append(nf.flows.LULinearPermute(dim))

        q0 = nf.distributions.DiagGaussian(dim, trainable=False)
        self.flow_model = nf.NormalizingFlow(q0=q0, flows=flows)

    def forward(self, z_0, x, sigma):
        """
        z_0, x: tensors of shape (batch, dim)
        sigma: shape (batch, 1) or scalar tensor
        returns: z, log_det
        """
        if sigma.dim() == 0:
            sigma = sigma.view(1, 1).expand(x.size(0), 1)
        elif sigma.dim() == 1:
            sigma = sigma.view(-1, 1)
        sigma = sigma.to(x.device)

        tcond = self.net1(sigma)                           # (batch, dim)
        h = self.net2(torch.cat([z_0, x], dim=1))          # (batch, dim)
        z0 = h + tcond                                     # (batch, dim)

        # In normflows, calling the module returns (z, log_det)
        z, log_det = self.flow_model.forward_and_log_det(z0)
        return z, log_det



class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, self.dim)  # w, u, b
        )

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z, sigma):

        log_jacobians = []
        sigma = sigma * torch.ones([z.shape[0],1]).to(z.device)

        tcond = self.net(sigma)
        z = z + tcond

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        return z, sum(log_jacobians)


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())
