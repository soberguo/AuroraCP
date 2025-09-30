import torch
import torch.nn.functional as F
from aurora import AuroraPretrained,AuroraSmallPretrained, Batch, Metadata,Aurora
import pickle
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from aurora.normalisation import locations, scales
from utils import hours_to_datetime,static_var,mean_std_1d,get_latest_checkpoint,mean_std_2d
import json
import torch.nn as nn
from model import CustomAurora
from weather_dataset import WeatherBench128, custom_collate
from torch.utils.data import Dataset, DataLoader
import os
from eval import evaluate
from args import get_args
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置可见的 GPU 设备
import torch.profiler


# class CustomModel(Aurora):
#     def __init__(self,model):
#         super(CustomModel, self).__init__()
#         self.model=model
#         self.surf_stats=self.model.surf_stats
#     def biaozhunhua(self,batch):
#         batch = self.batch_transform_hook(batch)
#         # Get the first parameter. We'll derive the data type and device from this parameter.
#         p = next(self.parameters())#[3,512]
#         batch = batch.type(p.dtype)
#         batch = batch.normalise(surf_stats=self.surf_stats)
#         return batch
#     def anti_biaozhunhua(self,batch):
#         batch =batch.unnormalise(surf_stats=self.surf_stats)
#         return batch

def main(args):
    dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=True)
    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate, batch_size=2,
        num_workers=16, pin_memory=True, drop_last=True)
    test_dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=False)
    test_loader = DataLoader(
            dataset=test_dataset,
            collate_fn=custom_collate, batch_size=1,
            num_workers=16, pin_memory=True, drop_last=True)
    static_vars_z, static_vars_lsm, static_vars_slt = static_var()
  
    surf_stats=mean_std_1d()

    model = AuroraPretrained(autocast=True,use_lora=True,stabilise_level_agg=True,
                            surf_vars=("2t", "10u", "10v",  "tp"),
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
        elif n.startswith('model.encoder.surf_token_embeds.weights') and ('10u' in n or '10v' in n or '2t' in n or 'tp' in n):
            p.requires_grad = True
        elif n.startswith('model.encoder.atmos_token_embeds.weights'):
            p.requires_grad = True
        elif 'surf_heads' in n and ('10u' in n or '10v' in n or '2t' in n or 'tp' in n):
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

    if args.resume:
        resume_result = get_latest_checkpoint(save_dir)
        if resume_result is not None:
            resume_path, start_epoch = resume_result
            print(f"🔁 加载最新断点: {resume_path}")
            checkpoint = torch.load(resume_path)
            CustomAuroraModel.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ 成功恢复到 epoch {start_epoch}")
        else:
            print("🆕 未找到断点，开始新训练")
    

    

    epochs=20
    lossmae=nn.L1Loss()
    lossmse=nn.MSELoss()
    for epoch in range(start_epoch,epochs):
        CustomAuroraModel.train()
        total_loss = 0.0 
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for i, (images, targets) in enumerate(train_iter):
            # profiler.step()  # 通知 profiler 进入下一个 step        
            images_1 = torch.stack([im[0].float() for im in images], dim=0).cuda()
            images_2 = torch.stack([im[1].float() for im in images], dim=0).cuda()
            target = torch.stack([t['tgt'].float() for t in targets],dim=0).cuda()
            if torch.isnan(images_1).any() or torch.isinf(images_1).any() or torch.isnan(images_2).any() or torch.isinf(images_2).any():
                # print("Input has NaN or Inf!")
                continue  # 跳过这个 batch，防止训练崩溃
            # target=guiyihua(target, surf_stats)
            # images_1=torch.stack(images_1, dim=0).cuda()
            # images_2=torch.stack(images_2, dim=0).cuda()
            var_2t=torch.stack([images_1[:, 0], images_2[:, 0]], dim=1) # [B, 2t, H, W]
            var_10u=torch.stack([images_1[:, 1], images_2[:, 1]], dim=1) # [B, 10u, H, W]
            var_10v=torch.stack([images_1[:, 2], images_2[:, 2]], dim=1)# [B, 10v, H, W]
            var_tp=torch.stack([images_1[:, 3], images_2[:, 3]], dim=1) # [B, msl, H, W]
            # var_z,var_u, var_v, var_t, var_q = [], [], [], [], []
            var_z =torch.stack([images_1[:, 4:17], images_2[:, 4:17]], dim=1)
            var_u =torch.stack([images_1[:, 17:30], images_2[:, 17:30]], dim=1)
            var_v =torch.stack([images_1[:, 30:43], images_2[:, 30:43]], dim=1)
            var_t =torch.stack([images_1[:, 43:56], images_2[:, 43:56]], dim=1) 
            var_r =torch.stack([images_1[:, 56:69], images_2[:, 56:69]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1) # [B, sshf, H, W]
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1) 
            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            batch=Batch(
                surf_vars={"2t": var_2t, "10u":var_10u, "10v":var_10v,"tp":var_tp,"sshf":var_sshf,"slhf":var_slhf},
                static_vars={"lsm":static_vars_lsm, "z":static_vars_z, "slt":static_vars_slt},
                atmos_vars={"z":var_z, "u":var_u, "v":var_v, "t":var_t, "r":var_r},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            # === forward + loss ===
            prediction = CustomAuroraModel(batch)
            target_batch=Batch(
                surf_vars={"2t": target[:, 0], "10u":target[:, 1], "10v":target[:, 2],"tp":target[:, 3]},
                static_vars={"lsm":static_vars_lsm, "z":static_vars_z, "slt":static_vars_slt},
                atmos_vars={"z":target[:, 4:17], "u":target[:, 17:30], "v":target[:, 30:43], "t":target[:, 43:56], "r":target[:, 56:69]},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            new_target=CustomAuroraModel.biaozhunhua(target_batch)

            mse_var_2t = F.mse_loss(prediction[1].surf_vars["2t"][:, 0], target[:, 0])
            mse_var_10u = F.mse_loss(prediction[1].surf_vars["10u"][:, 0], target[:, 1])
            mse_var_10v = F.mse_loss(prediction[1].surf_vars["10v"][:, 0], target[:, 2])
            mse_var_tp = F.mse_loss(prediction[1].surf_vars["tp"][:, 0], target[:, 3])
            mse_var_z = F.mse_loss(prediction[1].atmos_vars["z"][:, 0], target[:, 4:17])
            mse_var_u = F.mse_loss(prediction[1].atmos_vars["u"][:, 0], target[:, 17:30])
            mse_var_v = F.mse_loss(prediction[1].atmos_vars["v"][:, 0], target[:, 30:43])
            mse_var_t = F.mse_loss(prediction[1].atmos_vars["t"][:, 0], target[:, 43:56])
            mse_var_r = F.mse_loss(prediction[1].atmos_vars["r"][:, 0], target[:, 56:69])
            mse_var_z500 = F.mse_loss(prediction[1].atmos_vars["z"][:, 0][:,7,...], target[:,11])



            # mse_loss = (mse_var_2t + mse_var_10u + mse_var_10v + mse_var_z + mse_var_tp+
            #             mse_var_u + mse_var_v + mse_var_t + mse_var_r) / 9.0
            
            #---------------weight mae loss----------------
            mae_var_2t=lossmae(prediction[0].surf_vars["2t"][:, 0], new_target.surf_vars["2t"])
            mae_var_10u=lossmae(prediction[0].surf_vars["10u"][:, 0], new_target.surf_vars["10u"])
            mae_var_10v=lossmae(prediction[0].surf_vars["10v"][:, 0], new_target.surf_vars["10v"])
            mae_var_tp=lossmae(prediction[0].surf_vars["tp"][:, 0], new_target.surf_vars["tp"])
            # mae_var_sshf=lossmae(prediction[0].surf_vars["sshf"][:, 0], new_target.surf_vars["sshf"])
            # mae_var_slhf=lossmae(prediction[0].surf_vars["slhf"][:, 0], new_target.surf_vars["slhf"])
            surf_loss=(3.0)*mae_var_2t+(0.77)*mae_var_10u+(0.66)*mae_var_10v+(0.1)*mae_var_tp


            mae_var_z=lossmae(prediction[0].atmos_vars["z"][:, 0], new_target.atmos_vars["z"])
            mae_var_u=lossmae(prediction[0].atmos_vars["u"][:, 0], new_target.atmos_vars["u"])
            mae_var_v=lossmae(prediction[0].atmos_vars["v"][:, 0], new_target.atmos_vars["v"])
            mae_var_t=lossmae(prediction[0].atmos_vars["t"][:, 0], new_target.atmos_vars["t"])
            mae_var_r=lossmae(prediction[0].atmos_vars["r"][:, 0], new_target.atmos_vars["r"])
            atmos_loss=(2.8)*mae_var_z+(0.87)*mae_var_u+(0.6)*mae_var_v+(1.7)*mae_var_t+(0.78)*mae_var_r

            # reg_loss = - (lossmae(prediction[0].surf_vars["2t"][:, 0], prediction[2].surf_vars["2t"][:, 0]) + 
            #              lossmae(prediction[0].surf_vars["2t"][:, 0], prediction[2].surf_vars["2t"][:, 1])) 
            # penalty_weight = 0.01  # 可以调节
            # surf_loss=surf_loss+ penalty_weight * reg_loss
            mae_loss=(0.25*surf_loss+atmos_loss) *10
            #---------------weight mae loss---------------- 

            # === backward + optimize ===
            optim.zero_grad(set_to_none=True)
            mae_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"NaN gradient in {name}")
            optim.step()            
            total_loss += mae_loss.item()
            train_iter.set_postfix({"MAE loss":f"{mae_loss.item():.6f}",
                                    "rmse_2t":f"{mse_var_2t.item()**0.5:.6f}",
                                    "rmse_z500":f"{mse_var_z500.item()**0.5:.6f}",
                                    # "total_loss_with_reg":f"{total_loss_with_reg.item()**0.5:.6f}",
                                    # "mae_sshf":f"{mae_var_sshf.item()**0.5:.6f}",
                                    # "mae_slhf":f"{mae_var_slhf.item()**0.5:.6f}",
                                    })
            # if i >= 9:  # 只跑前 10 个 batch
            #     break
        # === 每个 epoch 输出 & lr 更新 ===
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{epochs}, Avg MAE Loss: {avg_loss:.6f}")
        lr_scheduler.step()
        # === 每 N epoch 保存模型 ===
        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': CustomAuroraModel.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, save_path)
            print(f"💾 模型已保存到: {save_path}")
        
        evaluate(CustomAuroraModel, test_loader)

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)