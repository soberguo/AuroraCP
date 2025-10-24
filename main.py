import torch
import torch.nn.functional as F
from aurora import AuroraPretrained, Batch, Metadata
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils import hours_to_datetime, static_var, mean_std_1d, get_latest_checkpoint
import torch.nn as nn
from model import CustomAurora
from weather_dataset import WeatherBench128, custom_collate
from torch.utils.data import DataLoader
import os
from eval import evaluate
from args import get_args
import logging
import sys

def setup_logging(log_file="training_log.txt"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

    
    


def main(args):
    logger = setup_logging("ori_training_log.txt")
    logger.info("🚀 开始训练 - 全面优化版本（显存优化 + NaN防护）")
    
    # ========== 优化2: 数据加载优化 ==========
    dataset = WeatherBench128(
        data_folder="/sharefiles2/guoyixin/datasets/new_weather_tensors2",
        n=6, train=True
    )
    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate,
        batch_size=args.batchsize,
        num_workers=4,
        pin_memory=False,  # 关闭pin_memory减少显存
        drop_last=True,
        prefetch_factor=2,  # 减少预加载
        persistent_workers=True  # 复用worker进程
    )
    
    test_dataset = WeatherBench128(
        data_folder="/sharefiles2/guoyixin/datasets/new_weather_tensors2",
        n=6, train=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=custom_collate,
        batch_size=1,
        num_workers=2,  # 减少测试时的worker
        pin_memory=False,
        drop_last=True
    )
    
    # ========== 优化3: 预先加载静态变量到GPU，避免重复传输 ==========
    static_vars_z, static_vars_lsm, static_vars_slt = static_var()
    # 移到GPU并设为不需要梯度
    static_vars_z = static_vars_z.cuda().requires_grad_(False)
    static_vars_lsm = static_vars_lsm.cuda().requires_grad_(False)
    static_vars_slt = static_vars_slt.cuda().requires_grad_(False)
    
    surf_stats = mean_std_1d()
    

    
    # ========== 模型初始化 ==========
    model = AuroraPretrained(
        autocast=True,
        use_lora=True,
        stabilise_level_agg=True,
        surf_vars=("2t", "10u", "10v", "tp","sshf","slhf"),
        atmos_vars=("z", "u", "v", "t", "r"),
        surf_stats=surf_stats
    )
    
    model.load_checkpoint_local('ckpt/aurora-0.25-pretrained.ckpt', strict=False)
    model = model.cuda()
    model.train()
    model.configure_activation_checkpointing()
    
    CustomAuroraModel = CustomAurora(model).cuda() 
    start_epoch = 0
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数设置
    for n, p in CustomAuroraModel.named_parameters():
        if 'lora_proj' in n or 'lora_qkv' in n:
            p.requires_grad = True
        elif n.startswith('model.encoder.surf_token_embeds.weights') and \
             ('10u' in n or '10v' in n or '2t' in n or 'tp' in n or 'sshf' in n or 'slhf' in n):
            p.requires_grad = True
        elif n.startswith('model.encoder.atmos_token_embeds.weights'):
            p.requires_grad = True
        elif 'surf_heads' in n and \
             ('10u' in n or '10v' in n or '2t' in n or 'tp' in n or 'sshf' in n or 'slhf' in n):
            p.requires_grad = True
        elif 'atmos_heads' in n:
            p.requires_grad = True
        elif 'hypernetwork' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    # ========== 优化5: 降低学习率，增加稳定性 ==========
    param_dicts = [
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (('lora_proj' in n and 'hypernetwork' not in n) or 
                                             ('lora_qkv' in n and 'hypernetwork' not in n))],
            "lr": 1e-3,  # 从1e-4降到5e-5
        },
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (n.startswith('model.encoder.surf_token_embeds.weights') or 
                                             n.startswith('model.encoder.atmos_token_embeds.weights') or 
                                             'surf_heads' in n or 'atmos_heads' in n or 'hypernetwork' in n)],
            "lr": 1e-4,  # 从1e-4降到5e-5
        },
    ]
    
    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters() if p.requires_grad)
    logger.info(f'可训练参数数量: {n_parameters:,}')
    
    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters())
    logger.info(f'总参数数量: {n_parameters:,}')
    
    # ========== 优化6: 优化器配置，添加eps防止除零 ==========
    optim = torch.optim.AdamW(
        param_dicts,
        lr=1e-4,  # 降低学习率
        weight_decay=1e-4,
        eps=1e-8,  # 添加eps防止除零
        betas=(0.9, 0.999)  # 使用标准beta
    )
    
    scaler = GradScaler(
        init_scale=2.**10,  # 初始缩放因子，不要太大
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
    
    # Resume训练
    if args.resume:
        resume_result = get_latest_checkpoint(save_dir)
        if resume_result is not None:
            resume_path, start_epoch = resume_result
            logger.info(f"🔁 加载最新断点: {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu')
            CustomAuroraModel.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"✅ 成功恢复到 epoch {start_epoch}")
        else:
            logger.info("🆕 未找到断点，开始新训练")
    
   
    
    # ========== 优化8: 梯度裁剪阈值 ==========
    max_grad_norm = 1.0
    
    # ========== 优化9: 预先创建常用张量，避免重复创建 ==========
    lat_tensor = torch.linspace(90, -90, 128).cuda()
    lon_tensor = torch.linspace(0, 360, 256 + 1)[:-1].cuda()
    atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    
    # 训练循环
    epochs = 20
    lossmae = nn.L1Loss()
    
    # ========== 优化10: 梯度累积（可选，如果显存还是不够） ==========
    accumulation_steps = getattr(args, 'accumulation_steps', 1)  # 默认不累积
    
    logger.info(f"训练配置: batch_size={args.batchsize}, accumulation_steps={accumulation_steps}, "
                f"effective_batch_size={args.batchsize * accumulation_steps}")
    
    for epoch in range(start_epoch, epochs):
        CustomAuroraModel.train()
        total_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for i, (images, targets) in enumerate(train_iter):
            # ========== 数据准备 ==========
            images_1 = torch.stack([im[0].float() for im in images], dim=0).cuda()
            images_2 = torch.stack([im[1].float() for im in images], dim=0).cuda()
            target = torch.stack([t['tgt'].float() for t in targets], dim=0).cuda()
            
            # ========== 优化11: 增强的数据检查 ==========
            if torch.isnan(images_1).any() or torch.isinf(images_1).any() or torch.isnan(images_2).any() or torch.isinf(images_2).any() or torch.isnan(target).any() or torch.isinf(target).any():
                # print("Input has NaN or Inf!")
                continue  # 跳过这个 batch，防止训练崩溃
            
           
            
            # ========== 构建变量（优化：及时删除不需要的） ==========
            var_2t = torch.stack([images_1[:, 0], images_2[:, 0]], dim=1)
            var_10u = torch.stack([images_1[:, 1], images_2[:, 1]], dim=1)
            var_10v = torch.stack([images_1[:, 2], images_2[:, 2]], dim=1)
            var_tp = torch.stack([images_1[:, 3], images_2[:, 3]], dim=1)
            var_z = torch.stack([images_1[:, 4:17], images_2[:, 4:17]], dim=1)
            var_u = torch.stack([images_1[:, 17:30], images_2[:, 17:30]], dim=1)
            var_v = torch.stack([images_1[:, 30:43], images_2[:, 30:43]], dim=1)
            var_t = torch.stack([images_1[:, 43:56], images_2[:, 43:56]], dim=1)
            var_r = torch.stack([images_1[:, 56:69], images_2[:, 56:69]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1)
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1)
            
            del images_1, images_2  # 及时释放
            
            time = tuple(hours_to_datetime(t['filename']) for t in targets)
            
            # ========== 优化12: 复用预先创建的张量 ==========
            batch = Batch(
                surf_vars={"2t": var_2t, "10u": var_10u, "10v": var_10v, 
                          "tp": var_tp, "sshf": var_sshf, "slhf": var_slhf},
                static_vars={"lsm": static_vars_lsm, "z": static_vars_z, "slt": static_vars_slt},
                atmos_vars={"z": var_z, "u": var_u, "v": var_v, "t": var_t, "r": var_r},
                metadata=Metadata(
                    lat=lat_tensor,
                    lon=lon_tensor,
                    time=time,
                    atmos_levels=atmos_levels,
                )
            )
            
            # ========== Forward + Loss (使用autocast) ==========
            with autocast():
                prediction = CustomAuroraModel(batch, args)
                
                # 计算target（不需要梯度）
                with torch.no_grad():
                    target_batch = Batch(
                        surf_vars={"2t": target[:, 0], "10u": target[:, 1], "10v": target[:, 2],
                                  "tp": target[:, 3], "sshf": target[:, 69], "slhf": target[:, 70]},
                        static_vars={"lsm": static_vars_lsm, "z": static_vars_z, "slt": static_vars_slt},
                        atmos_vars={"z": target[:, 4:17], "u": target[:, 17:30], "v": target[:, 30:43],
                                   "t": target[:, 43:56], "r": target[:, 56:69]},
                        metadata=Metadata(
                            lat=lat_tensor,
                            lon=lon_tensor,
                            time=time,
                            atmos_levels=atmos_levels,
                        )
                    )
                    new_target = CustomAuroraModel.biaozhunhua(target_batch)
                    del target_batch
                
                # ========== 优化14: 安全的loss计算 ==========
                # Surface loss
                mae_var_2t = lossmae(prediction[0].surf_vars["2t"][:, 0], new_target.surf_vars["2t"])
                mae_var_10u = lossmae(prediction[0].surf_vars["10u"][:, 0], new_target.surf_vars["10u"])
                mae_var_10v = lossmae(prediction[0].surf_vars["10v"][:, 0], new_target.surf_vars["10v"])
                mae_var_tp = lossmae(prediction[0].surf_vars["tp"][:, 0], new_target.surf_vars["tp"])
                
                # New variables loss
                mae_var_sshf = lossmae(prediction[0].surf_vars["sshf"][:, 0], new_target.surf_vars["sshf"])
                mae_var_slhf = lossmae(prediction[0].surf_vars["slhf"][:, 0], new_target.surf_vars["slhf"])
                # new_var_loss = mae_var_sshf + mae_var_slhf
                
                surf_loss = 3.0 * mae_var_2t + 0.77 * mae_var_10u + 0.66 * mae_var_10v + 0.1 * mae_var_tp+0.5*mae_var_sshf + 0.5*mae_var_slhf
                
                # Atmospheric loss
                mae_var_z = lossmae(prediction[0].atmos_vars["z"][:, 0], new_target.atmos_vars["z"])
                mae_var_u = lossmae(prediction[0].atmos_vars["u"][:, 0], new_target.atmos_vars["u"])
                mae_var_v = lossmae(prediction[0].atmos_vars["v"][:, 0], new_target.atmos_vars["v"])
                mae_var_t = lossmae(prediction[0].atmos_vars["t"][:, 0], new_target.atmos_vars["t"])
                mae_var_r = lossmae(prediction[0].atmos_vars["r"][:, 0], new_target.atmos_vars["r"])
                
                atmos_loss = (2.8 * mae_var_z + 0.87 * mae_var_u + 0.6 * mae_var_v + 
                             1.7 * mae_var_t + 0.78 * mae_var_r)
                
                # ========== 优化15: 降低loss缩放，配合梯度累积 ==========
                mae_loss = (0.25 * surf_loss + atmos_loss ) / accumulation_steps
                
                # Loss裁剪，防止异常
                if mae_loss > 50:
                    logger.warning(f"Loss过大: {mae_loss.item():.2f}，裁剪到50")
                    mae_loss = torch.clamp(mae_loss, max=50)

            
            # ========== Backward ==========
            scaler.scale(mae_loss).backward()
            
            # ========== 梯度累积 ==========
            if (i + 1) % accumulation_steps == 0:
                # ========== 优化17: 检查梯度 ==========
                
                # 梯度裁剪
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(CustomAuroraModel.parameters(), max_grad_norm)
                
                # 优化器更新
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            
            # ========== 监控指标（不需要梯度） ==========
            with torch.no_grad():
                mse_var_2t = F.mse_loss(prediction[1].surf_vars["2t"][:, 0], target[:, 0])
                mse_var_z500 = F.mse_loss(prediction[1].atmos_vars["z"][:, 0][:, 7, ...], target[:, 11])
            
            total_loss += mae_loss.item() * accumulation_steps
            train_iter.set_postfix({
                "MAE loss": f"{mae_loss.item() * accumulation_steps:.6f}",
                "rmse_2t": f"{mse_var_2t.item()**0.5:.6f}",
                "rmse_z500": f"{mse_var_z500.item()**0.5:.6f}",
                
            })
            
            # ========== 优化18: 及时清理显存 ==========
            del prediction, new_target, mae_loss, batch
            del var_2t, var_10u, var_10v, var_tp, var_z, var_u, var_v, var_t, var_r, var_sshf, var_slhf
            
            if i % 20 == 0:
                torch.cuda.empty_cache()
        
        # ========== Epoch结束 ==========
        avg_loss = total_loss / len(train_loader)
        
        lr_scheduler.step()
        
        # ========== 优化19: 保存更多信息到checkpoint ==========
        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': CustomAuroraModel.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'avg_loss': avg_loss,
            }, save_path)
            logger.info(f"💾 模型已保存到: {save_path}")
        
        # 评估
        torch.cuda.empty_cache()  # 评估前清理显存
        evaluate(CustomAuroraModel, test_loader, args, logger)
        torch.cuda.empty_cache()  # 评估后清理显存
    
    logger.info("🎉 训练完成!")


if __name__ == "__main__":
    args = get_args()
    logger = logging.getLogger()
    logger.info(f"训练参数: {args}")
    
    # 打印显存信息
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    
    main(args)
    