import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch.distributions import Normal
import numpy as np
from utils.model_utils import batch_KL_diag_gaussian_std

class TimeEmbedding(nn.Module):

    def __init__(self, t_embed_dim, scale=30.0):
        super().__init__()

        self.register_buffer("w", torch.randn(t_embed_dim//2)*scale)

    def forward(self, t):
        # t: (B, )
        t_proj = 2.0 * torch.pi * self.w[None, :] * t[:, None]  # (B, E//2)
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # (B, E)
        return t_embed

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = TimeEmbedding(num_out)

    def forward(self, x, y):
        out = self.lin(x)
        if y is not None:
            y=y.to(x.device)
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_out) + out
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, h_dim=400,learn_std=False):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, h_dim)
        self.lin2 = ConditionalLinear(h_dim, h_dim)
        self.lin3 = ConditionalLinear(h_dim, h_dim)
        self.lin4_mu = nn.Linear(h_dim, 2)
        self.learn_std=learn_std
        if learn_std:
            self.lin4_std = nn.Linear(h_dim, 2)

    
    def forward(self, x, y):
        # y=y.to(x.device)
        x = F.silu(self.lin1(x, y))
        x = F.silu(self.lin2(x, y))
        x = F.silu(self.lin3(x, y))
        x_mu = self.lin4_mu(x)
        if self.learn_std:
            x_std = torch.nn.functional.softplus(self.lin4_std(x))
            return x_mu, x_std
        return x_mu
    
class ScoreNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=400, learn_std=False):
        super(ScoreNet, self).__init__()
        self.learn_std = learn_std
        if learn_std:
            out_dim = input_dim * 2
        else:
            out_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, sigma=None):
        if sigma==None:
            sigma=torch.zeros_like(x)
        else:
            sigma = sigma.expand(-1, x.shape[-1])  
        x_concat = torch.cat([x, sigma], dim=-1)
        if self.learn_std:
            mu, std = self.net(x_concat).chunk(2, dim=-1)
            std = torch.nn.functional.softplus(std)
            return mu, std
        else:
            return self.net(x_concat)



def edm_sigma_schedule(Tmax, sigma_min=3.0, sigma_max=5, rho=7.0):
    device = 'cpu'  # or parameterize this
    step_indices = torch.arange(Tmax, device=device).float()
    t_steps = (sigma_max ** (1 / rho) +
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) *
               step_indices / (Tmax - 1)) ** rho
    return t_steps 
    

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def edm_noise_schedule(batch_size, device, sigma_min=0.002, sigma_max=80, rho =7.0):
    rand_t = torch.rand(batch_size, 1, device=device)  # Uniform [0,1]
    sigma = (sigma_max ** (1 / rho) + (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * rand_t) ** rho
    return sigma



def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)



class DiffusionModel(nn.Module):
    def __init__(self,num_steps=200, schedule='sigmoid', type='score', energy_func='sum', device="cpu", start=1e-5, end=1e-2):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.device=device

        scale=20.0
        betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=0.01, end=0.05).to(self.device)
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod)*0+1.0
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)*scale
        # sigma_list=one_minus_alphas_bar_sqrt*scale

        # self.betas = make_beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end).to(device)

        # self.alphas = 1 - self.betas
        # alphas_prod = torch.cumprod(self.alphas, 0)
        # self.alphas_bar_sqrt = torch.sqrt(alphas_prod)
        # self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        self.type=type

        if type=='score':
            self.score= ConditionalModel(num_steps)
        elif type=='denoise':
            self.mu= ConditionalModel(num_steps)
        elif type=='energy':
            if energy_func == "sum":
                self.net= ConditionalModel(num_steps)
                self.energy= lambda x,t: self.net(x,t).sum(-1)
                self.score= lambda x,t: -torch.autograd.grad(self.energy(x,t).sum(), x,create_graph=False)[0]
            else:
                raise NotImplementedError("energy_func must be 'sum'")
        else:
            raise NotImplementedError("type must be 'score' or 'energy'")


    def loss(self,x):
        batch_size = x.shape[0]
        t=torch.randint(0, self.num_steps,[batch_size], device=x.device)

        a = extract(self.alphas_bar_sqrt, t, x)
        sigma = extract(self.one_minus_alphas_bar_sqrt, t, x)
        e=torch.randn_like(x, device=x.device)
        xt = x * a + e * sigma
        if self.type=='score':
            score = self.score(xt, t)
        else:
            energy = self.energy(xt.requires_grad_(), t).sum()
            score =  -torch.autograd.grad(energy, xt,create_graph=True)[0]
        return torch.mean((e + sigma*score).square())
        # return torch.mean(torch.sum(2*e*sigma*score + sigma*score*sigma*score,-1)/a)
            

    def sample(self,num,snr=0.16,corrector=False):
        x = torch.randn([num,2], device=self.device)
        x_seq = [x]
        for t in reversed(range(self.num_steps)):
            t = torch.tensor([t], device=self.device)

            if corrector:
                if self.type=='score':
                    with torch.no_grad():
                        score = self.score(x, t)                
                elif self.type=='energy':
                    energy = self.energy(x.requires_grad_(), t).sum()
                    score =  -torch.autograd.grad(energy, x,create_graph=False)[0]
                else:
                    raise NotImplementedError("type must be 'score' or 'energy'")
                
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                lg_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + lg_step_size * score + torch.sqrt(2.0 * lg_step_size) * torch.randn_like(x, device=self.device)
            
            if self.type=='score':
                with torch.no_grad():
                    score = self.score(x, t)                
            elif self.type=='energy':
                energy = self.energy(x.requires_grad_(), t).sum()
                score =  -torch.autograd.grad(energy, x,create_graph=False)[0]
            else:
                raise NotImplementedError("type must be 'score' or 'energy'")
            
            x_mu=(1 / extract(self.alphas, t, x).sqrt())*(x+ extract(self.betas, t, x)*score)
            sigma_t = extract(self.betas, t, x).sqrt()
            x = x_mu + sigma_t * torch.randn_like(x, device=self.device)
            x_seq.append(x)
        return x
    

    
class network(nn.Module):
    def __init__(self,  input_dim=2, h_dim=400, output_dim=2, layer_num=4,learn_std=False):
        super().__init__()
        self.h_dim=h_dim
        self.input_dim=input_dim
        self.learn_std=learn_std
        self.acti=F.silu
        
        self.layers=nn.ModuleList()
        for i in range(layer_num-1):
            if i==0:
                self.layers.append(nn.Linear(input_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))

        if self.learn_std:
            self.out1 = nn.Linear(self.h_dim, output_dim)
            self.out2 = nn.Linear(self.h_dim, output_dim)
        else:
            self.out = nn.Linear(self.h_dim, output_dim)

    def forward(self,x):
        x=x.view(-1,self.input_dim)
        h=x
        for layer in self.layers:
            h=self.acti(layer(h))
        if self.learn_std:
            mu=self.out1(h)
            std=torch.nn.functional.softplus(self.out2(h))
            return mu, std
        else:
            return self.out(h)
            
class LVM(nn.Module):
    def __init__(self, z_dim,h_dim, x_dim, layer_num, boundary=None,device="cpu"):
        super().__init__()
        self.decoder = network(z_dim, h_dim, x_dim, layer_num, learn_std=False)
        self.prior_mu=torch.zeros(z_dim).to(device)
        self.prior_std=torch.ones(z_dim).to(device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.device=device
        self.boundary=boundary

    def sample(self, num):
        z = self.prior.sample([num]).to(self.device)
        x=self.decoder(z)
        x = x[~x.isnan().any(-1)]
        if self.boundary is not None:
            x = x[(x.abs()<=self.boundary).all(-1)]
        return x
    
class IVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = network(opt.x_dim+opt.z_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=False)
        self.decoder = network(opt.z_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False)
        self.prior_mu=torch.zeros(opt.z_dim).to(opt.device)
        self.prior_std=torch.ones(opt.z_dim).to(opt.device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.opt=opt

    def sample(self, num):
        z = torch.randn([num, self.opt.z_dim]).to(self.opt.device)
        samples=self.decoder(z)
        return samples.cpu()
    
class HSIVAE2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = network(opt.x_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.decoder = network(opt.z_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False)
        self.prior_mu=torch.zeros(opt.z_dim).to(opt.device)
        self.prior_std=torch.ones(opt.z_dim).to(opt.device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.opt=opt

    def sample(self, num):
        z = torch.randn([num, self.opt.z_dim]).to(self.opt.device)
        samples=self.decoder(z)
        return samples.cpu()

class HSIVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder_1 = network(opt.x_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.encoder_2 = network(opt.x_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.decoder = network(opt.z_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False)
        self.prior_mu=torch.zeros(opt.z_dim).to(opt.device)
        self.prior_std=torch.ones(opt.z_dim).to(opt.device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.opt=opt

    def sample(self, num):
        z = torch.randn([num, self.opt.z_dim]).to(self.opt.device)
        samples=self.decoder(z)
        return samples.cpu()
    
    
class SIVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = network(opt.x_dim+opt.z_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.decoder = network(opt.z_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False)
        self.prior_mu=torch.zeros(opt.z_dim).to(opt.device)
        self.prior_std=torch.ones(opt.z_dim).to(opt.device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.opt=opt

    def sample(self, num):
        z = torch.randn([num, self.opt.z_dim]).to(self.opt.device)
        samples=self.decoder(z)
        return samples.cpu()


class VAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = network(opt.x_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.decoder = network(opt.z_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False)
        self.prior_mu=torch.zeros(opt.z_dim).to(opt.device)
        self.prior_std=torch.ones(opt.z_dim).to(opt.device)
        self.prior=Normal(self.prior_mu, self.prior_std)  
        self.opt=opt

    def elbo(self,x):
        mu, std = self.encoder(x)
        z = mu + std*torch.randn_like(std)
        log_qz_giv_x=Normal(mu,std).log_prob(z).sum(1)
        log_pz=self.prior.log_prob(z).sum(1)
        x_mu=self.decoder(z)
        if self.opt.x_std is None:
            log_px_giv_z=-((x-x_mu)**2).sum(1)
        else:
            log_px_giv_z=Normal(x_mu,torch.ones_like(x_mu)*self.opt.x_std).log_prob(x).sum(1)
        return torch.mean(log_px_giv_z + log_pz - log_qz_giv_x)
    
    def spread_elbo(self,x,spread_std):
        tx_batch=x+torch.randn_like(x)*spread_std
        mu, std = self.encoder(tx_batch)
        z = mu + std*torch.randn_like(std)
        log_qz_giv_x=Normal(mu,std).log_prob(z).sum(1)
        log_pz=self.prior.log_prob(z).sum(1)
        x_mu=self.decoder(z)
        log_px_giv_z=Normal(x_mu,torch.ones_like(x_mu)*spread_std).log_prob(x).sum(1)
        return torch.mean(log_px_giv_z + log_pz - log_qz_giv_x)

    @torch.no_grad()
    def sample(self, num):
        z = torch.randn([num, self.opt.z_dim]).to(self.opt.device)
        samples=self.decoder(z)
        return samples.cpu()
    
    @torch.no_grad()
    def reconstruct(self, x):
        mu, std = self.encoder(x)
        z = mu + std*torch.randn_like(std)
        x_out = self.decoder(z)
        return x_out.cpu()


    
class AVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dims=[2]
        self.qa_x = network(self.dims[-1], opt.h_dim, self.dims[-1], opt.layer_num, learn_std=True)
        self.qz_a = network(self.dims[-1], opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True)
        self.qa_xz = network(self.dims[-1]*2, opt.h_dim, self.dims[-1], opt.layer_num, learn_std=True)
        self.px_z = network(opt.z_dim, opt.h_dim, self.dims[-1], opt.layer_num, learn_std=False)

        self.z_eps = None
        self.opt=opt
        self.pz_mu = torch.zeros(*self.dims).to(opt.device)
        self.pz_std = torch.ones(*self.dims).to(opt.device)

        # self.register_buffer('pz_mu', torch.zeros(*self.dims))
        # self.register_buffer('pz_std', torch.ones(*self.dims))

    def logpz(self,z):
        return Normal(self.pz_mu,self.pz_std).log_prob(z)
    


    def elbo(self,x):
        a_mu, a_std = self.qa_x(x)
        a=a_mu + a_std*torch.randn_like(a_std)
        z_mu, z_std= self.qz_a(a)
        z=z_mu + z_std*torch.randn_like(z_std)
        xz=torch.cat((x,z),dim=1)
        a_sm_mu, a_sm_std= self.qa_xz(xz)
        x_params=self.px_z(z)
        log_px_z=Normal(x_params,self.opt.x_std).log_prob(x).sum()
        kl_qa_x_pa_zx=batch_KL_diag_gaussian_std(a_mu,a_std,a_sm_mu, a_sm_std).sum()
        kl_qz_a_pz=batch_KL_diag_gaussian_std(z_mu,z_std,self.pz_mu,self.pz_std).sum()
        elbo=log_px_z-kl_qa_x_pa_zx-kl_qz_a_pz
        
        return (elbo/x.size(0))
    
    @torch.no_grad()
    def sample(self, num, fix_z=True):
        self.eval()
        if fix_z:
            if self.z_eps is None:
                self.z_eps = torch.randn(num, *self.dims).to(self.opt.device)
            z = self.z_eps
        else:
            z = torch.randn(num, *self.dims).to(self.opt.device)

        x_samples=self.px_z(z)
        return x_samples.cpu()
    
    @torch.no_grad()
    def reconstruct(self, x):
        a_mu, a_std = self.qa_x(x)
        a=a_mu + a_std*torch.randn_like(a_std)
        z_mu, z_std= self.qz_a(a)
        z=z_mu + z_std*torch.randn_like(z_std)
        x_params=self.px_z(z)
        return self.px_sample(x_params).cpu()
    
    @torch.no_grad()
    def z_pos_sample(self, x):
        a_mu, a_std = self.qa_x(x)
        a=a_mu + a_std*torch.randn_like(a_std)
        z_mu, z_std= self.qz_a(a)
        z=z_mu + z_std*torch.randn_like(z_std)
        return z.cpu()


