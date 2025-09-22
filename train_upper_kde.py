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
import time
from datetime import datetime

import yaml
import json
from torch.distributions import Normal

from evaluation import eval_log_prob
from evaluation import eval_mean_mmd_bandwidths as eval_mean_mmd

from utils.kde_mog import *

def sample_power_sigma(batch_size, power, min_val, max_val, device='cuda'):
    # Sample t uniformly from [0, 1]
    rand_t = torch.rand(batch_size, 1, device=device)
    sigma = min_val + torch.pow(rand_t,power) * (max_val - min_val)
    return sigma

def main(teacher_score_func, img_saver, true_samples, opt):
    log_prob_list = []
    MMD_list = []    
    time_list = []
    pow=0.1
    torch.manual_seed(opt.seed)


    global_start_time = time.time()
    #time_list.append(time0)
    
    diffusion_encoder=ScoreNet(learn_std=True).to(opt.device)
    decoder=ScoreNet(learn_std=False).to(opt.device)

    decoder_optim = optim.Adam(decoder.parameters(), lr=opt.lvm_lr)
    encoder_optim = optim.Adam(diffusion_encoder.parameters(), lr=opt.lvm_lr)

    with open(f"{opt.proj_path}/timing_log.txt", "a") as f:
        for it in tqdm(range(1, opt.max_iter + 1)):
            iter_start = time.perf_counter()  # High-res start timing

            half = opt.batch_size // 2
            z_half = torch.randn([half, opt.z_dim]).to(opt.device)
            z = torch.cat([z_half, z_half], dim=0)
            x0 = decoder(z)

            sigma = sample_power_sigma(half, power=pow, min_val=opt.sigma_min, max_val=opt.sigma_max, device=opt.device)
            sigma = torch.cat([sigma, sigma], dim=0)
            e_half = torch.randn_like(x0[:half])
            e = torch.cat([e_half, -e_half], dim=0)
            xt = x0 + e * sigma

            if opt.weight == "sigma":
                weight = (sigma / opt.sigma_max)
            elif opt.weight == "sigma2":
                weight = (sigma / opt.sigma_max) ** 2
            elif opt.weight == "sigma4":
                weight = (sigma / opt.sigma_max) ** 4
            else:
                weight = sigma * 0 + 1

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            z_mu, z_std = diffusion_encoder(xt, sigma)
            z_std = z_std * sigma
            logqz_xt = Normal(z_mu, z_std).log_prob(z).sum(-1)

            if opt.true_score:
                score_pd_xt = teacher_score_func(xt, sigma * 0 + 1, sigma).detach()
                mean_diff = score_pd_xt
            else:
                with torch.no_grad():
                    score_pd_xt = score_kde_2d(true_samples, xt, sigma)
                    mean_diff = score_pd_xt

            loss = -(weight * (logqz_xt + (mean_diff * x0).sum(-1))).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(diffusion_encoder.parameters(), max_norm=opt.grad_norm_clip)

            encoder_optim.step()
            decoder_optim.step()

            if it % opt.show_iter == 0:
                current_time = time.time()  # Wall clock time
                time_list.append(current_time-global_start_time)

                if pow < opt.power:
                    pow += 0.01
                    print(pow)

                decoder.eval()
                z = torch.randn([10000, opt.z_dim]).to(opt.device)
                x_samples = decoder(z)

                log_prob = eval_log_prob(
                    samples=x_samples,
                    mean_scale=torch.ones([10000, 1], device=opt.device),
                    std=torch.ones([10000, 1], device=opt.device) * torch.nn.functional.softplus(torch.tensor(1.0, device=opt.device))
                )
                log_prob_list.append(log_prob.item())

                # Compute mean MMD and append to list
                mean_mmd = eval_mean_mmd(true_samples, x_samples, opt.sigma_list)
                MMD_list.append(mean_mmd)

                print(f"Iteration {it}, mean MMD: {mean_mmd:.4f}, log prob: {log_prob:.4f}")

                with torch.no_grad():
                    img_saver(x_samples, save_name=f'{it},logprob={log_prob:.4f},MMD={mean_mmd:.4f}')
                    if it % (opt.show_iter *10)== 0:
                        torch.save(diffusion_encoder.state_dict(), f"{opt.proj_path}/model/encoder_{it}.pth")
                        torch.save(decoder.state_dict(), f"{opt.proj_path}/model/decoder_{it}.pth")

            # High-res time logging
            iter_end = time.perf_counter()
            elapsed = iter_end - iter_start
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Iteration {it}: {elapsed:.4f} seconds\n")


    with open(f'{opt.proj_path}/lists/logpd_list.json', 'w') as f:
        json.dump(log_prob_list, f)

    with open(f'{opt.proj_path}/lists/time_list.json', 'w') as f:
        json.dump(time_list, f)

    with open(f'{opt.proj_path}/lists/MMD_list.json', 'w') as f:
        json.dump(MMD_list, f)
        
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=40)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--weight", type=str, default='sigma2')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--name", type=str, default=None, help="Name of the run")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--true_score", action='store_true')

    args = parser.parse_args()

    if args.name is None:
        weight_str = args.weight if args.weight else "none"
        score_str = "true" if args.true_score else "learned"
        args.name = f"smin{args.sigma_min}_smax{args.sigma_max}_{weight_str}_{score_str}"

    print(args)

    opt=config(
        x_dim = 2,
        z_dim = 2,  
        h_dim = 400,
        layer_num = 5,
        batch_size=1024,
        show_iter = 1000,
        lvm_lr = 1e-4,
        boundary = 100,
        sigma_list = [2**-2, 2**-1, 2**0, 2**1, 2**2],
        grad_norm_clip = 10.0,
        #save
        save_path=f'./Gaussian_kde/',
        **vars(args)
    )


    if opt.load is not None:
        opt.load_path=f"""./save1/{opt.load}.pth"""


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
    os.makedirs(opt.proj_path+'/lists', exist_ok=True)

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
        
        true_samples = target.sample((10000,))
        log_prob = eval_log_prob(samples=true_samples.to(opt.device),
                                     mean_scale=torch.ones([10000,1],device=opt.device),
                                     std=torch.ones([10000,1],device=opt.device)*torch.nn.functional.softplus(torch.tensor(1.0,device=opt.device)))
        img_saver(true_samples,save_name=f'True Samples,log_prob={log_prob}')
        
        if opt.true_score:
            teacher_score_func=get_BatchGMM40_true_score
        else:
            from utils.edm_utils import edm_sample_2D, edm_sample_2D_kde
            teacher_score_func=ScoreNet(input_dim=2, hidden_dim=400).to(opt.device)
            # teacher_score_func.load_state_dict(torch.load('edm_score_network_max30.pth'))
            teacher_score_func.load_state_dict(torch.load('edm_network.pth'))
            print('----sampling from EDM diffusion with KDE score.')
            def kde_score_wrapper(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                return score_kde_2d(true_samples, x, sigma)
            # edm_samples = edm_sample_2D(teacher_score_func,num_samples=10000,num_steps=100,device=opt.device,sigma_max=40)
            kde_samples = edm_sample_2D_kde(
                                kde_score_wrapper,
                                device=opt.device,
                                num_samples=10000,
                                num_steps=100,
                                sigma_min=0.1,
                                sigma_max=40.0,
                                rho=7.0,
                                S_churn=0.0,     # you can try small churn like 0.1–0.3
                                S_min=0.0,
                                S_max=float("inf"),
                                S_noise=1.0
                            )
            log_prob = eval_log_prob(samples=kde_samples.to(opt.device),
                                     mean_scale=torch.ones([10000,1],device=opt.device),
                                     std=torch.ones([10000,1],device=opt.device)*torch.nn.functional.softplus(torch.tensor(1.0,device=opt.device)))
            mean_mmd = eval_mean_mmd(true_samples.to(opt.device), kde_samples.to(opt.device), opt.sigma_list)
            img_saver(kde_samples,save_name=f'kde samples,log_prob={log_prob},mmd={mean_mmd}')

    else:
        pass
    

    main(teacher_score_func, img_saver, true_samples, opt)
