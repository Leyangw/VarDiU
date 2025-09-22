import torch
from tqdm import tqdm
from model import *
import torch.optim as optim
from utils.mog_utils import *
from utils.model_utils import *
from types import SimpleNamespace as config
import wandb
import argparse
import os
import yaml
from torch.distributions import Normal
from math import exp
import copy

torch.manual_seed(0)


class EDM_schedule:
     def __init__(self, sigma_min=1.7, sigma_max=20, rho=7.0, device='cpu'):
         self.sigma_min = sigma_min
         self.sigma_max = sigma_max
         self.rho = rho
         self.device = device

     def sample_sigma(self, batch_size):
         # Sample t ~ Uniform[0, 1]
         rand_t = torch.rand(batch_size, 1, device=self.device)

         # Continuous EDM noise schedule
         sigma = (self.sigma_max ** (1 / self.rho) +
                  (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)) * rand_t) ** self.rho
         return sigma # Shape: [B, 1]


def main(teacher_score_func, img_saver, opt):    
    anneal_num = opt.start_anneal
    
    diffusion_encoder = ScoreNet(learn_std=True).to(opt.device)
    decoder = ScoreNet(learn_std=False).to(opt.device)

    #if not opt.true_score:
    #    decoder = copy.deepcopy(teacher_score_func).train().requires_grad_(True).to(opt.device)

    decoder_optim = optim.Adam(decoder.parameters(), lr=opt.lvm_lr)
    encoder_optim = optim.Adam(diffusion_encoder.parameters(), lr=opt.lvm_lr)        

    for it in tqdm(range(1,opt.max_iter+1)):        
        half = opt.batch_size // 2

        z=torch.randn([half, opt.z_dim]).to(opt.device)
        z=torch.cat([z, z], dim=0)
        x0=decoder(x=z,sigma=torch.ones([opt.batch_size,1],device=opt.device))   

        if opt.anneal:
            rho = exp(anneal_num)
        else:
            rho=opt.rho

        sigma = EDM_schedule(sigma_min=opt.sigma_min, sigma_max=opt.sigma_max, rho=rho,device=opt.device).sample_sigma(half)
        sigma = torch.cat([sigma, sigma], dim=0)
        e = torch.randn_like(x0[:half])
        e = torch.cat([e, -e], dim=0)
        xt = x0 + e * sigma
            
        if opt.weight=="sigma":
            weight=(sigma/opt.sigma_max)
        elif opt.weight=="quad":
            weight=(sigma/opt.sigma_max)**4
        else:
            weight=sigma*0+1

        for _ in range(0,opt.e_step):
            encoder_optim.zero_grad()
            z_mu, z_std = diffusion_encoder(xt.detach(), sigma)
            z_std=z_std*sigma
            logqz_xt = weight*Normal(z_mu, z_std).log_prob(z).sum(-1)
            loss=-logqz_xt.mean()
            loss.backward()
            encoder_optim.step()

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        z_mu, z_std = diffusion_encoder(xt, sigma)
        z_std=z_std*sigma
        logqz_xt = Normal(z_mu, z_std).log_prob(z).sum(-1)

        if opt.true_score:
            score_pd_xt=teacher_score_func(xt, sigma*0+1, sigma).detach()
            mean_diff=score_pd_xt 
            loss = -(weight*(logqz_xt+ (mean_diff.detach()*xt).sum(-1))).mean()
        else:
            with torch.no_grad():
                x0_mu=teacher_score_func(xt, sigma)
                mean_diff=(x0_mu - x0)/sigma**2     
            loss = -(weight*(logqz_xt + (mean_diff.detach()*x0).sum(-1))).mean()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=opt.grad_norm_clip)
        encoder_optim.step()
        decoder_optim.step()

        if it%(opt.show_iter)==0:
            if opt.anneal and anneal_num < opt.max_num:
                anneal_num += 0.01
                print(f'current rho:{exp(anneal_num)}')

            decoder.eval()
            z=torch.randn([10000, opt.z_dim]).to(opt.device)
            x_samples= decoder(x=z,sigma=torch.ones([z.shape[0],1],device=opt.device)) 
            with torch.no_grad():
                img_saver(x_samples,save_name=it)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--e_step", type=int, default=0)
    parser.add_argument("--max_num", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=20.0)
    parser.add_argument("--weight", type=str, default='quad')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--anneal", action='store_true')
    parser.add_argument("--start_anneal", type=float, default=-2)
    parser.add_argument("--name", type=str, default='upper_diffusion', help="Name of the run")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--true_score", action='store_true')

    args = parser.parse_args()

    print(args)

    opt=config(
        x_dim = 2,
        z_dim = 2,  
        h_dim = 400,
        layer_num = 5,
        batch_size=1024,
        max_iter = 1000000,
        show_iter = 1000,
        lvm_lr = 1e-4,
        boundary = 100,
        
        grad_norm_clip = 10.0,
        #save
        save_path='./save/',
        **vars(args)
    )

    
    if opt.load is not None:
        opt.load_path=f"""./save/{opt.load}.pth"""


    run = wandb.init(
        mode="online" if opt.wandb else "offline",
        project='spread',
        name=args.name,
        config=dict(vars(opt))
    )
    

        
    opt.proj_path=opt.save_path+opt.name
    if os.path.exists(opt.proj_path):
        counter = 1
        # Loop until a non-existing path is found
        while os.path.exists(opt.proj_path):
            opt.proj_path = opt.save_path + opt.name + f"_{counter}"
            counter += 1

    os.makedirs(opt.proj_path, exist_ok=True)
    os.makedirs(opt.proj_path+'/model', exist_ok=True)
    os.makedirs(opt.proj_path+'/plot', exist_ok=True)
    with open(opt.proj_path+'/config.yaml', 'w') as file:
        yaml.dump(vars(opt), file)
        
    if opt.target=="mog9":
        target=MoG9(dim=2,std=0.5,device=opt.device)
        # target.to(opt.device)
        # teacher_score_func=get_BatchMoG9_true_score
        img_saver=lambda x,save_name: plot_MoG9(
            samples=x, 
            save_path=opt.proj_path+'/plot',
            save_name=save_name,
        )
        
        
    elif opt.target=="mog40":
        target = GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device=opt.device)
        target.to(opt.device)
        img_saver=lambda x,save_name: plot_MoG40(
            log_prob_function=GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device="cpu").log_prob,
            samples=x, 
            save_path=opt.proj_path+'/plot',
            save_name=save_name,
            title=None
            )
        
        if opt.true_score:
            teacher_score_func=get_BatchGMM40_true_score
        else:
            teacher_score_func=ScoreNet(input_dim=2, hidden_dim=400).to(opt.device)
            # teacher_score_func.load_state_dict(torch.load('edm_score_network_max30.pth'))
            teacher_score_func.load_state_dict(torch.load('edm_network.pth'))

            # teacher_score_model=DiffusionModel(
            #         num_steps=100, 
            #         schedule='linear', 
            #         type='score',
            #         start=0.01, 
            #         end=0.05,
            #         device=opt.device,
            #         ).to(opt.device)
            # teacher_score_model.load_state_dict(torch.load('save/dm_1/model/score_model.pth'))
            # teacher_score_func=teacher_score_model.score

        

    else:
        pass
    

    main(teacher_score_func, img_saver,opt)
