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


torch.manual_seed(0)

def main(teacher_score_func, img_saver, opt):
    data=target.sample([10000])   
    img_saver(data,save_name=-1)

    num_steps = opt.n_steps
    betas = make_beta_schedule(schedule=opt.schedule, n_timesteps=opt.n_steps, start=0.0001, end=0.05).to(opt.device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)*0+1
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)*3
    print(one_minus_alphas_bar_sqrt)
    # print(one_minus_alphas_bar_sqrt)
    
    diffusion_encoder=ConditionalModel(opt.n_steps, opt.h_dim, learn_std=True).to(opt.device)

    decoder=network(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False).to(opt.device)

    decoder_optim = optim.Adam(decoder.parameters(), lr=opt.lvm_lr)
    encoder_optim = optim.Adam(diffusion_encoder.parameters(), lr=opt.lvm_lr)

    for it in tqdm(range(opt.max_iter)):
        ### E step
        for _ in range(opt.e_steps):
            with torch.no_grad():
                z=torch.randn([opt.batch_size, opt.z_dim]).to(opt.device)
                x0=decoder(z)
                t=torch.randint(0, num_steps,[opt.batch_size], device=x0.device)
                a = extract(alphas_bar_sqrt, t, x0)
                sigma = extract(one_minus_alphas_bar_sqrt, t, x0)
                e=torch.randn_like(x0, device=x0.device)
                xt = x0 * a + e * sigma

            encoder_optim.zero_grad()
            z_mu, z_std = diffusion_encoder(xt, t)
            loss = -Normal(z_mu, z_std).log_prob(z).sum(-1).mean()
            loss.backward()
            encoder_optim.step()
        
        ### M step
        decoder_optim.zero_grad()
        z=torch.randn([opt.batch_size, opt.z_dim]).to(opt.device)
        logpz=Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
        x0=decoder(z)
        t=torch.randint(0, num_steps,[opt.batch_size], device=x0.device)
        a = extract(alphas_bar_sqrt, t, x0)
        sigma = extract(one_minus_alphas_bar_sqrt, t, x0)
        e=torch.randn_like(x0, device=x0.device)
        xt = x0 * a + e * sigma
        logpxt_z = Normal(a*x0,sigma).log_prob(xt).sum(-1)
        z_mu, z_std = diffusion_encoder(xt, t)
        logqz_xt = Normal(z_mu, z_std).log_prob(z).sum(-1)
        score_pd_xt=teacher_score_func(xt, a, sigma).detach()
        loss = (logpxt_z + logpz - logqz_xt - (score_pd_xt*xt).sum(-1)).mean()
        loss.backward()
        decoder_optim.step()

        if it%opt.show_iter==0:
            decoder.eval()
            z=torch.randn([10000, opt.z_dim]).to(opt.device)
            x_samples= decoder(z)
            with torch.no_grad():
                img_saver(x_samples,save_name=it)

            torch.save(x_samples, f'{opt.proj_path}/plot/upper_samples.pt')
            torch.save(decoder.state_dict(), f'{opt.proj_path}/plot/upper_decoder.pth')
            torch.save(diffusion_encoder.state_dict(), f'{opt.proj_path}/plot/upper_encoder.pth')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--e_steps", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--x_std", type=float, default=0.5) 
    parser.add_argument("--temp", type=float, default=0.1) 
    parser.add_argument("--name", type=str, default='upper_anneal', help="Name of the run")
    parser.add_argument("--device", type=str, default='cpu')
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
    
        teacher_score_func=get_BatchGMM40_true_score
    else:
        pass
    

    main(teacher_score_func, img_saver,opt)