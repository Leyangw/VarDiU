import torch
import torch.optim as optim
from torch.distributions import Normal

from tqdm import tqdm
from model import *
from utils.mog_utils import *
from utils.model_utils import *
from types import SimpleNamespace as config
import wandb
import argparse
import os
import yaml

torch.manual_seed(0)

def main(target, img_saver, opt):
    data=target.sample([10000])   
    img_saver(data,save_name=-1)

    lvm = LVM(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, opt.boundary,opt.device).to(opt.device).to(opt.device)
    
    encoder=network(opt.x_dim, opt.h_dim, opt.z_dim, opt.layer_num, learn_std=True).to(opt.device)
    lvm_optim = optim.Adam(lvm.parameters(), lr=opt.lvm_lr)
    encoder_optim = optim.Adam(encoder.parameters(), lr=opt.lvm_lr)

    for it in tqdm(range(opt.max_iter)):
        encoder_optim.zero_grad()
        lvm_optim.zero_grad()
        z=lvm.prior.sample([opt.batch_size]).to(opt.device)
        x_mu=lvm.decoder(z)
        x = x_mu+torch.randn_like(x_mu)*opt.x_std
        logpz=lvm.prior.log_prob(z).sum(-1)
        logpxz=Normal(x_mu,opt.x_std).log_prob(x).sum(-1)
        z_mu, z_std = encoder(x)
        logqz_x = Normal(z_mu, z_std).log_prob(z).sum(-1)
        logpx= target.log_prob(x)
        loss = (logpxz + logpz - logqz_x - logpx).mean()
        # print(loss.item())
        loss.backward()
        wandb.log({
            'loss':loss.item(),
        },step=it)
        ## clip
        # torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip) 
        encoder_optim.step()       
        lvm_optim.step()
        
        if it%opt.show_iter==0:
            lvm.eval()
            x_mu= lvm.sample(10000)
            saved_samples = x_mu+torch.randn_like(x_mu)*opt.x_std
            # z=lvm.prior.sample([10000]).to(opt.device)
            # x_mu,x_std=decoder(z)
            # saved_samples = Normal(x_mu,x_std).sample().detach().cpu()
            with torch.no_grad():
                img_saver(saved_samples,save_name=it)

            torch.save(saved_samples, f'{opt.proj_path}/plot/upper_samples.pt')
            torch.save(lvm.state_dict(), f'{opt.proj_path}/plot/upper_lvm.pth')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--x_std", type=float, default=0.5) 
    parser.add_argument("--temp", type=float, default=0.1) 
    parser.add_argument("--name", type=str, default='upper_anneal', help="Name of the run")
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--anneal", action='store_true')

    args = parser.parse_args()

    print(args)

    opt=config(
        x_dim = 2,
        z_dim = 2,  
        h_dim = 400,
        layer_num = 5,
        batch_size=1024,
        max_iter = 100000,
        show_iter = 1000,
        lvm_lr = 1e-4,
        boundary = 100,
        
        grad_norm_clip = 1.0,
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
    
        # teacher_score_func=get_BatchGMM40_true_score
    else:
        pass
    

    main(target, img_saver,opt)