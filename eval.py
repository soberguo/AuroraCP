import torch
from tqdm import tqdm
import torch.nn.functional as F
from aurora import AuroraPretrained,AuroraSmallPretrained, Batch, Metadata,rollout
from utils import hours_to_datetime
from utils import static_var,mean_std_1d,hours_to_datetime,save_visualization,mean_std_2d
from weather_dataset import WeatherBench128, custom_collate
from torch.utils.data import Dataset, DataLoader
from model import CustomAurora
from aurora.normalisation import locations, scales
from args import get_args
import os

@torch.no_grad()
def evaluate(model, test_loader,args, logger,device="cuda"):
    static_vars_z, static_vars_lsm, static_vars_slt = static_var()
    model.eval()
    total_loss = 0.0

    # 初始化各变量的 loss 累加器
    loss_dict = {
        "2t": 0.0, "10u": 0.0, "10v": 0.0,"tp": 0.0,
        "z": 0.0, "u": 0.0, "v": 0.0,
        "t": 0.0, "r": 0.0,
        "sshf":0.0,"slhf":0.0
    }
    nan_num=0   
    with torch.no_grad():
        for (images, targets) in tqdm(test_loader, desc="Evaluating", leave=False):
            images_1 = torch.stack([im[0].float() for im in images], dim=0).to(device)
            images_2 = torch.stack([im[1] for im in images], dim=0).to(device)
            target=torch.stack([t['tgt'] for t in targets],dim=0).cuda()
            if torch.isnan(images_1).any() or torch.isinf(images_1).any() or torch.isnan(images_2).any() or torch.isinf(images_2).any() or torch.isnan(target).any() or torch.isinf(target).any():
                nan_num+=1
                # print("Input has NaN or Inf!")
                continue  # 跳过这个 batch，防止训练崩溃
            var_2t = torch.stack([images_1[:, 0], images_2[:, 0]], dim=1)
            var_10u = torch.stack([images_1[:, 1], images_2[:, 1]], dim=1)
            var_10v = torch.stack([images_1[:, 2], images_2[:, 2]], dim=1)
            var_tp = torch.stack([images_1[:, 3], images_2[:, 3]], dim=1)
            var_z = torch.stack([images_1[:, 4:17], images_2[:, 4:17]], dim=1)
            var_u = torch.stack([images_1[:, 17:30], images_2[:, 17:30]], dim=1)
            var_v = torch.stack([images_1[:, 30:43], images_2[:, 30:43]], dim=1)
            var_t = torch.stack([images_1[:, 43:56], images_2[:, 43:56]], dim=1)
            var_r = torch.stack([images_1[:, 56:69], images_2[:, 56:69]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1)  # [B, sshf, H, W]
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1)
            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            batch=Batch(
                surf_vars={"2t": var_2t, "10u":var_10u, "10v":var_10v,"tp":var_tp, "sshf":var_sshf,"slhf":var_slhf},
                static_vars={"lsm":static_vars_lsm, "z":static_vars_z, "slt":static_vars_slt},
                atmos_vars={"z":var_z, "u":var_u, "v":var_v, "t":var_t, "r":var_r},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            with torch.inference_mode():
                prediction = [pred.to("cpu") for pred in rollout(model, batch, steps=args.roll_step,args=args)]

            # prediction = model(batch,args)
            
            # 每类变量 MSE
            loss_dict["2t"] += F.mse_loss(prediction[1].surf_vars["2t"][:, 0], target[:, 0]).item()
            loss_dict["10u"] += F.mse_loss(prediction[1].surf_vars["10u"][:, 0], target[:, 1]).item()
            loss_dict["10v"] += F.mse_loss(prediction[1].surf_vars["10v"][:, 0], target[:, 2]).item()
            loss_dict["tp"] += F.mse_loss(prediction[1].surf_vars["tp"][:, 0], target[:, 3]).item()
            loss_dict["z"] += F.mse_loss(prediction[1].atmos_vars["z"][:, 0], target[:, 4:17]).item()
            loss_dict["u"] += F.mse_loss(prediction[1].atmos_vars["u"][:, 0], target[:, 17:30]).item()
            loss_dict["v"] += F.mse_loss(prediction[1].atmos_vars["v"][:, 0], target[:, 30:43]).item()
            loss_dict["t"] += F.mse_loss(prediction[1].atmos_vars["t"][:, 0], target[:, 43:56]).item()
            loss_dict["r"] += F.mse_loss(prediction[1].atmos_vars["r"][:, 0], target[:, 56:69]).item()
            loss_dict["sshf"] += F.mse_loss(prediction[1].surf_vars["sshf"][:, 0], target[:, 69]).item()
            # # if torch.isnan(torch.tensor(loss_dict["sshf"])).any():
            # #     print("SSHf loss is NaN!")
            loss_dict["slhf"] += F.mse_loss(prediction[1].surf_vars["slhf"][:, 0], target[:, 70]).item()

    # 平均化每项 loss
    num_batches = len(test_loader)-nan_num
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict.items()}
    avg_total_loss = sum(avg_loss_dict.values()) / len(avg_loss_dict)

    # 打印每项 loss
    print("\n📊 Evaluation Results:")
    for var, loss in avg_loss_dict.items():
        
        if logger==None:
            print(f"  - RMSE [{var:>4}]: {loss**0.5:.12f}")
        else:
            logger.info(f"  - RMSE [{var:>4}]: {loss**0.5:.12f}")
    if logger==None:
        print(f"✅ Avg Total Eval RMSE : {avg_total_loss**0.5:.6f}\n")
    else:
        logger.info(f"Avg Total Eval RMSE : {avg_total_loss**0.5:.6f}\n")
    return avg_total_loss 


if __name__ == "__main__":
    args = get_args()
    print(args)
    surf_stats=mean_std_1d()
    static_vars_z, static_vars_lsm, static_vars_slt = static_var()

    model = AuroraPretrained(autocast=True,use_lora=True,stabilise_level_agg=True,
                            surf_vars=("2t", "10u", "10v",  "tp","sshf","slhf"),
                            atmos_vars=("z", "u", "v", "t",  "r"),
                            surf_stats=surf_stats
                            )
    

    model.load_checkpoint_local('ckpt/aurora-0.25-pretrained.ckpt',strict=False)
    model=model.cuda()
    model.train()
    model.configure_activation_checkpointing()


    CustomAuroraModel = CustomAurora(model).cuda()
    start_epoch = 0
    save_dir = "./aurora_checkpoints"
    os.makedirs(save_dir, exist_ok=True)


    for n, p in CustomAuroraModel.named_parameters():
        if 'lora_proj' in n or 'lora_qkv' in n:
            p.requires_grad = True
            # print(n)
        elif n.startswith('model.encoder.surf_token_embeds.weights') and ('10u' in n or '10v' in n or '2t' in n or 'tp' in n or 'sshf' in n or 'slhf' in n):
            p.requires_grad = True
        elif n.startswith('model.encoder.atmos_token_embeds.weights'):
            p.requires_grad = True
        elif 'surf_heads' in n and ('10u' in n or '10v' in n or '2t' in n or 'tp' in n or 'sshf' in n or 'slhf' in n):
            p.requires_grad = True
        elif 'atmos_heads' in n:
            p.requires_grad = True
        elif 'hypernetwork' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

        
    param_dicts = [
        {
        "params": [p for n, p in CustomAuroraModel.named_parameters()
                if p.requires_grad and (('lora_proj' in n and 'hypernetwork' not in n) or ('lora_qkv' in n and 'hypernetwork' not in n))]},
        { ## others
        "params": [p for n, p in CustomAuroraModel.named_parameters()
                if p.requires_grad and (n.startswith('model.encoder.surf_token_embeds.weights') or 
                                        n.startswith('model.encoder.atmos_token_embeds.weights') or 
                                        'surf_heads' in n or 'atmos_heads' in n
                                        or 'hypernetwork' in n)],
        "lr": 1e-3,
        },]

    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters() if p.requires_grad)
    print('number of leanable params:', n_parameters)

    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters())
    print('number of all params:', n_parameters)
    optim = torch.optim.AdamW(
        param_dicts, lr=3e-4,
        weight_decay=1e-4
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10)

    
    print(f"🔁 加载最新断点: {args.ckpt}")
    checkpoint = torch.load(args.ckpt)
    CustomAuroraModel.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # print(f"✅ 成功恢复到 epoch {start_epoch}")
        
    CustomAuroraModel.eval()

    test_dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=False,roll_step=args.roll_step)
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=4, pin_memory=False, drop_last=True)
    # 假设 test_loader 已经定义并加载了测试数据
    # 例如: test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # evaluate 函数会打印每个变量的 MSE loss 和平均总 loss
    logger=None
    evaluate(CustomAuroraModel, test_loader,args,logger)  # 请确保 test_loader 已定义并包含测试数据