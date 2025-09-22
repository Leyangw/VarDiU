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


torch.manual_seed(0)


def main(target, img_saver, opt):
    
    lvm = LVM(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, opt.boundary,opt.device).to(opt.device)
    cool=network(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False).to(opt.device)
    cool_gen=lambda x: x+cool(x)
    lvm.load_state_dict(torch.load('./save/upper_anneal_1/plot/upper_lvm.pth'))
    cool_optim = optim.Adam(cool.parameters(), lr=opt.lvm_lr)
    score_model=network(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, learn_std=False).to(opt.device)

    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)

    for it in tqdm(range(1,opt.max_iter+1)):
        cool.eval()
        score_model.train()

        for _ in range(opt.score_iter):
            score_optim.zero_grad()
            with torch.no_grad():
                x_hot=lvm.sample(opt.batch_size)
                x=cool_gen(x_hot)
            
            score_loss = aapo_sm_loss(lambda x: score_model(x), x)

            score_loss.backward()
            score_optim.step()

        cool.train()
        score_model.eval()
        cool_optim.zero_grad()
        
        with torch.no_grad():
            x_hot=lvm.sample(opt.batch_size)
        x=cool_gen(x_hot)

        target_score = torch.func.grad(lambda y: target.log_prob(y).sum())(x).detach()

        lvm_score=score_model(x)
        if it%1000==0:
            opt.temp+=0.1
            opt.temp=min(opt.temp,1)

        score_diff = (lvm_score - opt.temp*target_score).detach()
        lvm_loss=(score_diff*x).sum(1).mean()

        if ~(torch.isnan(lvm_loss) | torch.isinf(lvm_loss)):
            lvm_loss.backward()

        torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip)
        cool_optim.step()


        if it==1 or it%opt.show_iter==0:
            lvm.eval()
            x_hot=lvm.sample(10000)
            saved_samples =cool_gen(x_hot).cpu()
            with torch.no_grad():
                img_saver(saved_samples,save_name=it)

            # torch.save(saved_samples, f'{opt.proj_path}/plot/peking_samples_{it}.pt')
            # torch.save(lvm.state_dict(), f'{opt.proj_path}/plot/peking_lvm_{it}.pth')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--score_loss", type=str, default='sm')
    parser.add_argument("--score_iter", type=int, default=50)
    parser.add_argument("--Tmin", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.2) 
    parser.add_argument("--score_type", type=str, default='score')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--name", type=str, default='rkl_cool', help="Name of the run")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    print(args)

    opt=config(
        x_dim = 2,
        h_dim = 400,
        layer_num = 5,
        max_iter = 100000,
        show_iter = 1000,
        lvm_lr = 1e-3,
        Tmax = 1,
        start = 1e-4,
        end = 0.7,
        dsm_scheme='linear',
        score_lr = 1e-4,

        batch_size = 1024,
        boundary = 100,

        num_langevin_steps = 5,
        langevin_step_size = 1e-2,
        n_is = 10,
        AIS_step = 15,
        hmc_step_size = 1.0,
        
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
        target=MoG9(dim=2,device=opt.device)
        target.to(opt.device)
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