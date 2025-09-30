from weather_dataset import WeatherBench128, custom_collate
from torch.utils.data import Dataset, DataLoader
from utils import hours_to_datetime,mean_std_1d,get_latest_checkpoint
import torch
from small_model.small_model import Small_model
from tqdm import tqdm
from aurora import Batch,Metadata
import torch.nn as nn
import os
import torch.nn.functional as F

def small_eval(model, test_loader):
    
    model.eval()
    total_loss = 0.0

    # 初始化各变量的 loss 累加器
    loss_dict = {
        "2t": 0.0, "sshf": 0.0,"slhf": 0.0
    }

    with torch.no_grad():
        for (images, targets) in tqdm(test_loader, desc="Evaluating", leave=False):
            images_1 = torch.stack([im[0].float() for im in images], dim=0).cuda()
            images_2 = torch.stack([im[1] for im in images], dim=0).cuda()
            target=torch.stack([t['tgt'] for t in targets],dim=0).cuda()

            var_2t = torch.stack([images_1[:, 0], images_2[:, 0]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1)  # [B, sshf, H, W]
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1)

            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            batch=Batch(
                surf_vars={"2t": var_2t, "sshf":var_sshf, "slhf":var_slhf},
                static_vars={},
                atmos_vars={},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))

            prediction = model(batch)
            preds=Batch(
                surf_vars={"2t": prediction},
                static_vars={},
                atmos_vars={},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            prediction=model.anti_biaozhunhua(preds)
            
            # 每类变量 MSE
            loss_dict["2t"] += F.mse_loss(prediction.surf_vars["2t"], target[:, 0]).item()
            
            # loss_dict["sshf"] += F.mse_loss(prediction[1].surf_vars["sshf"][:, 0], target[:, 69]).item()
            # loss_dict["slhf"] += F.mse_loss(prediction[1].surf_vars["slhf"][:, 0], target[:, 70]).item()

    # 平均化每项 loss
    num_batches = len(test_loader)
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict.items()}
    avg_total_loss = sum(avg_loss_dict.values()) / len(avg_loss_dict)

    # 打印每项 loss
    print("\n📊 Evaluation Results:")
    for var, loss in avg_loss_dict.items():
        print(f"  - RMSE [{var:>4}]: {loss**0.5:.12f}")
    print(f"✅ Avg Total Eval RMSE : {avg_total_loss**0.5:.6f}\n")

    return avg_total_loss 

def main():
    dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=True)
    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate, batch_size=4,
        num_workers=4, pin_memory=False, drop_last=True)
    test_dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=custom_collate, batch_size=10,
        num_workers=4, pin_memory=False, drop_last=True)
    surf_stats=mean_std_1d()
    model=Small_model(surf_stats=surf_stats).cuda().train()
    start_epoch = 0
    epochs=20
    save_dir = "./checkpoints"

    for n, p in model.named_parameters():
        p.requires_grad = True

    param_dicts = [
        {
        "params": [p for n, p in model.named_parameters()
                if p.requires_grad]},
        ]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of leanable params:', n_parameters)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of all params:', n_parameters)
    optim = torch.optim.AdamW(
        param_dicts, lr=3e-4,
        weight_decay=1e-4
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10)


    resume = False
    if resume:
        resume_result = get_latest_checkpoint(save_dir)
        if resume_result is not None:
            resume_path, start_epoch = resume_result
            print(f"🔁 加载最新断点: {resume_path}")
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ 成功恢复到 epoch {start_epoch}")
        else:
            print("🆕 未找到断点，开始新训练")


    epochs=20
    lossmae=nn.L1Loss()
    for epoch in range(start_epoch,epochs):
        model.train()
        total_loss = 0.0 
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for i, (images, targets) in enumerate(train_iter):

            images_1 = torch.stack([im[0].float() for im in images], dim=0).cuda()
            images_2 = torch.stack([im[1].float() for im in images], dim=0).cuda()
            target = torch.stack([t['tgt'].float() for t in targets],dim=0).cuda()
            if torch.isnan(images_1).any() or torch.isinf(images_1).any() or torch.isnan(images_2).any() or torch.isinf(images_2).any():
                # print("Input has NaN or Inf!")
                continue  # 跳过这个 batch，防止训练崩溃
            var_2t=torch.stack([images_1[:, 0], images_2[:, 0]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1) # [B, sshf, H, W]
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1) 
            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            batch=Batch(
                surf_vars={"2t": var_2t, "sshf":var_sshf, "slhf":var_slhf},
                static_vars={},
                atmos_vars={},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            preds=model(batch)
            target_batch=Batch(
                surf_vars={"2t": target[:, 0],},
                static_vars={},
                atmos_vars={},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            new_target=model.biaozhunhua(target_batch)
            mae_loss=lossmae(preds, new_target.surf_vars["2t"])
            optim.zero_grad(set_to_none=True)
            mae_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"NaN gradient in {name}")
            optim.step()            
            total_loss += mae_loss.item()
            train_iter.set_postfix({"MAE loss":f"{mae_loss.item():.6f}",
                                    # "rmse_2t":f"{mse_var_2t.item()**0.5:.6f}",
                                    # "rmse_z500":f"{mse_var_z500.item()**0.5:.6f}",
                                    # "total_loss_with_reg":f"{total_loss_with_reg.item()**0.5:.6f}",
                                    # "mae_sshf":f"{mae_var_sshf.item()**0.5:.6f}",
                                    # "mae_slhf":f"{mae_var_slhf.item()**0.5:.6f}",
                                    })

        # === 每个 epoch 输出 & lr 更新 ===
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{epochs}, Avg MAE Loss: {avg_loss:.6f}")
        lr_scheduler.step()
        # === 每 N epoch 保存模型 ===
        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, save_path)
            print(f"💾 模型已保存到: {save_path}")
        small_eval(model, test_loader)


if __name__ == "__main__":
    main()

    # test_dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=False)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     collate_fn=custom_collate, batch_size=10,
    #     num_workers=4, pin_memory=False, drop_last=True)
    # surf_stats=mean_std_1d()
    # model=Small_model(surf_stats=surf_stats).cuda().train()
    # resume_path='checkpoints/epoch_001.pt'
    # checkpoint = torch.load(resume_path)  # 或 'model.pth'
    # print(f"🔁 加载最新断点: {resume_path}")
    # for n, p in model.named_parameters():
    #     p.requires_grad = True

    # param_dicts = [
    #     {
    #     "params": [p for n, p in model.named_parameters()
    #             if p.requires_grad]},
    #     ]
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of leanable params:', n_parameters)

    # n_parameters = sum(p.numel() for p in model.parameters())
    # print('number of all params:', n_parameters)
    # optim = torch.optim.AdamW(
    #     param_dicts, lr=3e-4,
    #     weight_decay=1e-4
    #     )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # small_eval(model, test_loader)
