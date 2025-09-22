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


def main(teacher_score_func,img_saver,opt):

    lvm = LVM(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, opt.boundary,opt.device).to(opt.device)
    lvm_optim = optim.Adam(lvm.parameters(), lr=opt.lvm_lr)

    score_model= DiffusionModel(
        num_steps=opt.Tmax, 
        schedule=opt.dsm_scheme, 
        type=opt.score_type,
        start=opt.start, 
        end=opt.end,
        device=opt.device,
        ).to(opt.device)
    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)

    for it in tqdm(range(1,opt.max_iter+1)):
        lvm.eval()
        score_model.train()

        for _ in range(opt.score_iter):
            score_optim.zero_grad()
            x=lvm.sample(opt.batch_size).detach()
            score_loss  = score_model.loss(x)
            if opt.score_beta>0:
                sm_loss = aapo_sm_loss(lambda x: score_model.score(x,torch.zeros([x.shape[0]], device=opt.device)), x)
                score_loss += opt.score_beta * sm_loss
            score_loss.backward()
            score_optim.step()

        lvm.train()
        score_model.eval()
        lvm_optim.zero_grad()
        x = lvm.sample(opt.batch_size)

        t = torch.randint(opt.Tmin, score_model.num_steps,[x.shape[0]], device=x.device)
        a = extract(score_model.alphas_bar_sqrt.to(opt.device), t, x)
        sigma = extract(score_model.one_minus_alphas_bar_sqrt.to(opt.device), t, x)
        e=torch.randn_like(x, device=x.device)
        x_t = (x * a + e * sigma)
        lvm_score=score_model.score(x_t,t)
        teacher_score=teacher_score_func(x_t,a,sigma)
        
        if opt.weight=="uniform":
            w=1
        elif opt.weight=="1/a":
            w=1/a
        elif opt.weight=="linear_anneal":
            w=(it/opt.max_iter)+(1-it/opt.max_iter)*(1/a)
        elif opt.weight=="poly_anneal":
            eta = (it / opt.max_iter) ** opt.poly_coeff 
            w = eta * 1 + (1 - eta) * (1 / a)
        else:
            raise 

            
        score_diff = w*(lvm_score - teacher_score).detach()
        lvm_loss=(score_diff*x_t).sum(1).mean()

        if ~(torch.isnan(lvm_loss) | torch.isinf(lvm_loss)):
            lvm_loss.backward()

        torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip)
        lvm_optim.step()


        if it==1 or it%opt.show_iter==0:
            lvm.eval()
            with torch.no_grad():
                img_saver(lvm.sample(10000).cpu(),save_name=it)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--target", type=str, default='mog40')
    parser.add_argument("--score_loss", type=str, default='sm')
    parser.add_argument("--weight", type=str, default='linear_anneal')
    parser.add_argument("--poly_coeff", type=float, default=2)
    parser.add_argument("--score_beta", type=float, default=0.0)
    parser.add_argument("--score_iter", type=int, default=50)
    parser.add_argument("--Tmin", type=int, default=1)
    parser.add_argument("--score_type", type=str, default='score')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--name", type=str, default='rkl_gt_score', help="Name of the run")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    print(args)

    opt=config(
    x_dim = 2,
    h_dim = 400,
    layer_num = 5,
    max_iter = 50000,
    show_iter = 1000,
    lvm_lr = 1e-3,

    Tmax = 30,
    start = 1e-4,
    end = 0.7,
    dsm_scheme='linear',
    score_lr = 1e-4,

    batch_size = 1024,
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
        # target=MoG9(dim=2,device=opt.device)
        teacher_score_func=get_BatchMoG9_true_score
        img_saver=lambda x,save_name: plot_MoG9(
            samples=x, 
            save_path=opt.proj_path+'/plot',
            save_name=save_name,
        )
        
        
    elif opt.target=="mog40":
        target = GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device="cpu")
        img_saver=lambda x,save_name: plot_MoG40(
            log_prob_function=target.log_prob,
            samples=x, 
            save_path=opt.proj_path+'/plot',
            save_name=save_name,
            title=None
            )
    
        teacher_score_func=get_BatchGMM40_true_score
    else:
        pass
    

    main(teacher_score_func,img_saver,opt)