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
import copy
import json
import time

from evaluation import eval_mean_mmd_bandwidths as eval_mean_mmd
from evaluation import eval_log_prob

class EDM_loss:
    def __init__(self,opt,P_mean=-1.2,P_std=1.2,sigma_data=10,device='cpu'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.device = device
        self.opt = opt

    def __call__(self, net, y):
        #rnd_normal = torch.randn([y.shape[0], 1], device=y.device)
        #sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = sample_power_sigma(y.shape[0],power=self.opt.power,min_val=self.opt.sigma_min,max_val=self.opt.sigma_max,device=self.opt.device)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


def sample_power_sigma(batch_size, power, min_val, max_val, device='cuda'):
    # Sample t uniformly from [0, 1]
    rand_t = torch.rand(batch_size, 1, device=device)
    sigma = min_val + torch.pow(rand_t,power) * (max_val - min_val)
    return sigma


def main(teacher_score_func, img_saver, opt):    
    torch.manual_seed(opt.seed)

    log_prob_list = []
    MMD_list = []    
    time_list = []
    pow = opt.power
    global_start_time = time.time()

    student_score = ScoreNet(learn_std=False).to(opt.device)

    if not opt.true_score:
        teacher_score_func.eval().requires_grad_(False)
        student_score = copy.deepcopy(teacher_score_func).train().requires_grad_(True).to(opt.device)
    else:
        student_score = copy.deepcopy(teacher_score).train().requires_grad_(True).to(opt.device)

    decoder = ScoreNet(learn_std=False).to(opt.device)

    #decoder = copy.deepcopy(teacher_score_func).train().requires_grad_(True).to(opt.device)

    decoder_optim = optim.Adam(decoder.parameters(), lr=opt.g_lr)
    student_optim = optim.Adam(student_score.parameters(), lr=opt.score_lr)   

    for it in tqdm(range(1,opt.max_iter+1)):       
        score_loss = EDM_loss(opt=opt,sigma_data=10,device=opt.device)

        #train student score
        student_score.train().requires_grad_(True)
        decoder.eval().requires_grad_(False)

        for _ in range(0,opt.score_step):
            with torch.no_grad():
                z=torch.randn([opt.batch_size, opt.z_dim]).to(opt.device)
                x0=decoder(x=z,sigma=torch.ones([opt.batch_size,1],device=opt.device))

            loss = score_loss(student_score,x0.detach())

            student_optim.zero_grad()
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(student_score.parameters(), max_norm=opt.grad_norm_clip)
            student_optim.step()

        student_optim.zero_grad()
        decoder_optim.zero_grad()

        student_score.eval().requires_grad_(False)
        decoder.train().requires_grad_(True)

        #train decoder
        half = opt.batch_size // 2
        z_half=torch.randn([half, opt.z_dim]).to(opt.device)
        z=torch.cat([z_half, z_half], dim=0)
        x0=decoder(z)        
        
        sigma = sample_power_sigma(half,power=pow,min_val=opt.sigma_min,max_val=opt.sigma_max,device=opt.device)
        sigma = torch.cat([sigma,sigma],dim=0)
        e_half = torch.randn_like(x0[:half])
        e = torch.cat([e_half, -e_half], dim=0)
        xt = x0 + e * sigma

        if opt.weight=="sigma":
            weight=(sigma/opt.sigma_max)
        elif opt.weight=="sigma2":
            weight=(sigma/opt.sigma_max)**2
        elif opt.weight=="sigma4":
            weight=(sigma/opt.sigma_max)**4
        else:
            weight=sigma*0+1

        if not opt.true_score:
            with torch.no_grad():
                x0_teacher=teacher_score_func(xt, sigma)
                x0_student = student_score(xt, sigma)
                mean_diff = (x0_student - x0_teacher) / sigma**2   
        else:
            x0_teacher=teacher_score_func(xt, sigma*0+1, sigma).detach() * sigma**2 + xt
            with torch.no_grad():
                x0_student = student_score(xt, sigma)
                mean_diff = (x0_student - x0_teacher) / sigma**2   

        loss = weight * mean_diff.detach() * x0
        loss = loss.mean()

        decoder_optim.zero_grad()

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=opt.grad_norm_clip)

        decoder_optim.step()

        if it%(opt.show_iter)==0:
            current_time = time.time()  # Wall clock time
            time_list.append(current_time - global_start_time)
            
            decoder.eval()
            z=torch.randn([10000, opt.z_dim]).to(opt.device)
            x_samples= decoder(x=z,sigma=torch.ones([z.shape[0],1],device=opt.device)) 
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
                img_saver(x_samples,save_name=f'{it},logprob={log_prob},mmd={mean_mmd}')

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
    parser.add_argument("--score_step", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=20.0)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--weight", type=str, default='sigma2')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--name", type=str, default='DiKL', help="Name of the run")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--true_score", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(args)

    opt=config(
        x_dim = 2,
        z_dim = 2,  
        h_dim = 400,
        layer_num = 5,
        batch_size=1024,
        show_iter = 1000,
        score_lr = 5e-5,
        g_lr = 1e-4,
        boundary = 100,
        sigma_list = [2**-2, 2**-1, 2**0, 2**1, 2**2],
        #grad clip
        grad_norm_clip = 10.0,
        #save
        save_path='./DiKL_step1_fake/',
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
            teacher_score = ScoreNet(input_dim=2, hidden_dim=400).to(opt.device)
            # teacher_score_func.load_state_dict(torch.load('edm_score_network_max30.pth'))
            teacher_score.load_state_dict(torch.load('edm_network.pth'))
            teacher_score.eval().requires_grad_(False)
        else:
            teacher_score_func=ScoreNet(input_dim=2, hidden_dim=400).to(opt.device)
            # teacher_score_func.load_state_dict(torch.load('edm_score_network_max30.pth'))
            teacher_score_func.load_state_dict(torch.load('edm_network.pth'))

    else:
        pass
    

    main(teacher_score_func, img_saver,opt)