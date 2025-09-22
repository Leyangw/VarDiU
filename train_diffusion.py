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


def main(train_data,img_saver,opt):
    train_size=train_data.size(0)

    score_model= DiffusionModel(
        num_steps=opt.Tmax, 
        schedule=opt.dsm_scheme, 
        type=opt.score_type,
        start=0.01, 
        end=0.05,
        device=opt.device,
        ).to(opt.device)
    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)

    
    loss_list=[]
    for it in tqdm(range(1,opt.max_iter+1)):
        score_model.train()
        score_optim.zero_grad()
        x= train_data[torch.randint(0,train_size,[opt.batch_size],device=opt.device)]
        score_loss  = score_model.loss(x)
        score_loss.backward()
        score_optim.step()
        loss_list.append(score_loss.item())

        
        if it%opt.show_iter==0:
            score_model.eval()
            with torch.no_grad():
                # img_saver(score_model.sample(10000),save_name=it)
                # plt.plot(loss_list)
                # plt.savefig(opt.proj_path+f'''/plot/loss_{it}.png''')
                # plt.close()

                # Save the dictionary
                torch.save(score_model.state_dict(), f"""{opt.proj_path}/model/score_model.pth""")

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")
    parser.add_argument("--score_type", type=str, default='score')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--name", type=str, default='dm', help="Name of the run")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    print(args)

    opt=config(
    max_iter = 500000,
    show_iter = 10000,

    Tmin = 0,
    Tmax = 100,
    start = 1e-5,
    end = 1e-2,
    dsm_scheme='linear',
    score_lr = 1e-3,

    batch_size = 1024,
    boundary = 100,
    
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
        
    # target=MoG9(dim=2,device=opt.device)
    # train_data = target.sample([50000])
    # img_saver=lambda x,save_name: plot_MoG9(
    #     samples=x, 
    #     save_path=opt.proj_path+'/plot',
    #     save_name=save_name,
    # )

    target = GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device="cpu")
    img_saver=lambda x,save_name: plot_MoG40(
            log_prob_function=target.log_prob,
            samples=x, 
            save_path=opt.proj_path+'/plot',
            save_name=save_name,
            title=None
            )
    
    teacher_score_function=lambda x,t: target.score(x,t)
    
    train_samples = target.sample([50000]).to(opt.device)
    # test_samples = target.sample([10000])

    main(train_samples,img_saver,opt)
