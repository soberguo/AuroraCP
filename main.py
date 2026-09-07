import torch
import torch.nn.functional as F
from aurora import AuroraPretrained, Batch, Metadata,rollout
from torch.amp import GradScaler
from tqdm import tqdm
from utils import hours_to_datetime, static_var, compute_loss, construct_batch, get_latest_checkpoint, build_batch_from_tensor
import torch.nn as nn
from model import CustomAurora
from data.weather_dataset import WeatherBench2, custom_collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from eval import evaluate
from args import get_args
from distributed_utils import (
    barrier,
    cleanup_distributed,
    distributed_all_true,
    distributed_mean,
    initialize_distributed,
    seed_everything,
    wrap_ddp,
)
import logging
from datetime import timedelta
import sys

def setup_logging(log_file="training_log.txt", log_dir="log", is_main=True, rank=0):
    """
    设置日志，保存到指定文件夹

    Args:
        log_file: 日志文件名
        log_dir: 日志文件夹路径
    """
    # 创建日志文件夹
    logger = logging.getLogger(f"aurora.main.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if not is_main:
        logger.addHandler(logging.NullHandler())
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
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




def main(args, logger, context):
    device = context.device


    # ========== 优化2: 数据加载优化 ==========
    dataset = WeatherBench2(
        data_folder=args.data_folder,
        roll_step=0,timestep=args.timestep,
        years=args.train_year,args=args
    )
    train_sampler = None
    if context.distributed:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=args.shuffle_train,
            seed=args.seed,
            drop_last=args.drop_last,
        )
    train_loader = DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        collate_fn=custom_collate,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        shuffle=args.shuffle_train if train_sampler is None else False
    )

    test_loader = None
    if context.is_main:
        test_dataset = WeatherBench2(
            data_folder=args.data_folder,
            roll_step=(args.roll_step-1),timestep=args.timestep,years=args.test_year,args=args,
            num_rollout_targets=args.roll_step if args.roll_step > 1 else 0,
            return_train_aux_target=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            collate_fn=custom_collate,
            batch_size=1,
            num_workers=args.test_num_workers,
            pin_memory=args.test_pin_memory,
            drop_last=args.drop_last
        )

    # ========== 优化3: 预先加载静态变量到GPU，避免重复传输 ==========
    static_vars_z, static_vars_lsm, static_vars_slt = static_var(
        static_path=args.static_path,
        target_size=(args.target_height, args.target_width),
    )
    # 移到GPU并设为不需要梯度
    args.static_vars_z = static_vars_z.to(device).requires_grad_(False)
    args.static_vars_lsm = static_vars_lsm.to(device).requires_grad_(False)
    args.static_vars_slt = static_vars_slt.to(device).requires_grad_(False)

    # surf_stats = mean_std_1d()



    # ========== 模型初始化 ==========
    model = AuroraPretrained(
        autocast=args.use_amp,
        use_lora=args.use_lora,
        stabilise_level_agg=args.stabilise_level_agg,
        timestep=timedelta(hours=args.timestep),
        surf_vars=tuple(args.surf_vars),
        atmos_vars=tuple(args.atmos_vars),
    )

    model.load_checkpoint_local(args.pretrained, strict=False)
    model = model.to(device)
    model.train()
    if args.activation_checkpointing:
        model.configure_activation_checkpointing()

    CustomAuroraModel = CustomAurora(
        model,
        timestep=timedelta(hours=args.timestep),
        land_vars=tuple(args.land_vars),
        target_height=args.target_height,
        target_width=args.target_width,
        atmos_levels=tuple(args.atmos_levels),
    ).to(device)
    start_epoch = 0
    save_dir = args.save_dir
    if context.is_main:
        os.makedirs(save_dir, exist_ok=True)
    barrier(context)



    for n, p in CustomAuroraModel.named_parameters():
        if 'lora_proj' in n or 'lora_qkv' in n:
            p.requires_grad = True
        elif n.startswith('model.encoder.surf_token_embeds.weights') and \
             any(var in n for var in args.surf_vars):
            p.requires_grad = True
        elif n.startswith('model.encoder.atmos_token_embeds.weights'):
            p.requires_grad = True
        elif 'surf_heads' in n and \
             any(var in n for var in args.surf_vars):
            p.requires_grad = True
        elif 'atmos_heads' in n:
            p.requires_grad = True
        elif 'hypernetwork' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ========== 优化5: 为不同参数组设置学习率 ==========
    param_dicts = [
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (('lora_proj' in n and 'hypernetwork' not in n) or
                                             ('lora_qkv' in n and 'hypernetwork' not in n))],
            "lr": args.lr1,
        },
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (n.startswith('model.encoder.surf_token_embeds.weights') or
                                             n.startswith('model.encoder.atmos_token_embeds.weights') or
                                             'surf_heads' in n or 'atmos_heads' in n or 'hypernetwork' in n)],
            "lr": args.lr2,
        },
    ]

    # ========== 打印所有可训练参数 ==========
    logger.info("=" * 80)
    logger.info("可训练参数详情:")
    logger.info("=" * 80)

    # 按类别统计
    category_counts = {}
    for n, p in CustomAuroraModel.named_parameters():
        if not p.requires_grad:
            continue
        if 'hypernetwork' in n:
            cat = 'hypernetwork'
        elif 'lora_proj' in n or 'lora_qkv' in n:
            cat = 'lora (main model)'
        elif n.startswith('model.encoder.surf_token_embeds.weights'):
            cat = 'surf_token_embeds'
        elif n.startswith('model.encoder.atmos_token_embeds.weights'):
            cat = 'atmos_token_embeds'
        elif 'surf_heads' in n:
            cat = 'surf_heads'
        elif 'atmos_heads' in n:
            cat = 'atmos_heads'
        elif 'norm' in n:
            cat = 'norm (main model)'
        else:
            cat = 'other'
        category_counts[cat] = category_counts.get(cat, 0) + p.numel()

    for cat in category_counts:
        logger.info(f"  [{cat}]: {category_counts[cat]:,} params")

    # 确定每个参数属于哪个优化器group
    group1_params = [p for _n, p in CustomAuroraModel.named_parameters()
                     if p.requires_grad and (('lora_proj' in _n and 'hypernetwork' not in _n) or
                                            ('lora_qkv' in _n and 'hypernetwork' not in _n))]
    group2_params = [p for _n, p in CustomAuroraModel.named_parameters()
                     if p.requires_grad and (_n.startswith('model.encoder.surf_token_embeds.weights') or
                                            _n.startswith('model.encoder.atmos_token_embeds.weights') or
                                            'surf_heads' in _n or 'atmos_heads' in _n or 'hypernetwork' in _n)]
    group1_set = set(id(p) for p in group1_params)
    group2_set = set(id(p) for p in group2_params)
    group_default_params = [p for p in (p for _n, p in CustomAuroraModel.named_parameters())
                            if p.requires_grad and id(p) not in group1_set and id(p) not in group2_set]

    logger.info("-" * 80)
    logger.info(f"优化器参数组:")
    logger.info(f"  Group 1 (LoRA, lr={args.lr1}): {sum(p.numel() for p in group1_params):,} params")
    logger.info(f"  Group 2 (embeds/heads/hypernetwork, lr={args.lr2}): {sum(p.numel() for p in group2_params):,} params")
    logger.info(f"  Default  (norm等, lr={args.base_lr}): {sum(p.numel() for p in group_default_params):,} params")
    logger.info("=" * 80)

    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters() if p.requires_grad)
    logger.info(f'可训练参数数量: {n_parameters:,}')


    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters())
    logger.info(f'总参数数量: {n_parameters:,}')

    # ========== 优化6: 优化器配置，添加eps防止除零 ==========
    optim = torch.optim.AdamW(
        param_dicts,
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
        betas=tuple(args.adam_betas)
    )

    scaler = GradScaler(
        device.type,
        init_scale=2.**10,  # 初始缩放因子，不要太大
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    # Resume训练：传入 --ckpt 即恢复，不传则从 pretrained 初始化
    if args.ckpt:
        resume_path= args.ckpt
        logger.info(f"🔁 加载最新断点: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        CustomAuroraModel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.reset_lr_on_resume:
            configured_lrs = (args.lr1, args.lr2)
            if len(optim.param_groups) != len(configured_lrs):
                raise ValueError(
                    "无法重置学习率: optimizer 参数组数量 "
                    f"{len(optim.param_groups)} 与配置数量 {len(configured_lrs)} 不一致"
                )
            for param_group, configured_lr in zip(optim.param_groups, configured_lrs):
                param_group['lr'] = configured_lr
                param_group['initial_lr'] = configured_lr
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optim,
                step_size=args.scheduler_step_size,
                gamma=args.scheduler_gamma,
            )
            logger.info(
                "阶段切换后重置学习率和 scheduler: "
                f"lr1={args.lr1}, lr2={args.lr2}"
            )
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"✅ 成功恢复到 epoch {start_epoch}")

    training_model = wrap_ddp(
        CustomAuroraModel,
        context,
        find_unused_parameters=args.ddp_find_unused_parameters,
    )




    # ========== 优化8: 梯度裁剪阈值 ==========
    max_grad_norm = args.max_grad_norm

    # ========== 优化9: 预先创建常用张量，避免重复创建 ==========
    args.lat_tensor = torch.linspace(90, -90, args.target_height + 1)[:-1].to(device)
    args.lon_tensor = torch.linspace(0, 360, args.target_width + 1)[:-1].to(device)
    args.atmos_levels = tuple(args.atmos_levels)

    # 训练循环
    epochs = args.epochs


    # ========== 优化10: 梯度累积（可选，如果显存还是不够） ==========
    accumulation_steps = getattr(args, 'accumulation_steps', 1)  # 默认不累积
    monitor_surf_var = args.main_surf_vars[0] if args.main_surf_vars else args.surf_vars[0]
    monitor_atmos_var = args.atmos_vars[0]

    logger.info(f"训练配置: batch_size_per_gpu={args.batchsize}, accumulation_steps={accumulation_steps}, "
                f"world_size={context.world_size}, "
                f"effective_batch_size={args.batchsize * accumulation_steps * context.world_size}")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        total_loss = 0.0
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            disable=not context.is_main,
        )

        for i, (images, targets) in enumerate(train_iter):
            # ========== 数据准备 ==========
            images_1 = torch.stack([im[0].float() for im in images], dim=0).to(device)
            images_2 = torch.stack([im[1].float() for im in images], dim=0).to(device)
            target = torch.stack([t['tgt'].float() for t in targets], dim=0).to(device)
            if args.train_roll_step!=1:
                target2 = torch.stack([t['tgt2'].float() for t in targets], dim=0).to(device)

            # ========== 优化11: 增强的数据检查 ==========
            tensors_to_validate = [images_1, images_2, target]
            if args.train_roll_step != 1:
                tensors_to_validate.append(target2)
            local_batch_valid = not any(
                torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()
                for tensor in tensors_to_validate
            )
            if not distributed_all_true(local_batch_valid, context):
                continue  # 跳过这个 batch，防止训练崩溃



            time = tuple(hours_to_datetime(t['filename']) for t in targets)

            # ========== 优化12: 复用预先创建的张量 ==========
            batch = build_batch_from_tensor(images_2, time, args, history_tensor=images_1)
            del images_1, images_2

            # ========== Forward + Loss (使用autocast) ==========

            #单步训练
            if args.train_roll_step==1:
                prediction = training_model(batch, args)
            else:
                predictions = [pred for pred in rollout(training_model, batch, steps=args.train_roll_step,args=args)]
                prediction=predictions[0]
                pred2=predictions[1]
                pred1=CustomAuroraModel.biaozhunhua(prediction)
                pred2=CustomAuroraModel.biaozhunhua(pred2)
            # 计算target（不需要梯度）
            with torch.no_grad():
                target_batch=construct_batch(target,time,args)

                new_target = CustomAuroraModel.biaozhunhua(target_batch)
                if args.train_roll_step!=1:
                    target_batch2=construct_batch(target2,time,args)
                    new_target2 = CustomAuroraModel.biaozhunhua(target_batch2)

            if args.train_roll_step==1:
                mae_loss=compute_loss(prediction[0], new_target, args)
            else:
                mae_loss1=compute_loss(pred1, new_target, args)
                mae_loss2=compute_loss(pred2, new_target2, args)
                mae_loss=mae_loss1+mae_loss2


            # Loss裁剪，防止异常
            if args.loss_clip_max > 0 and mae_loss > args.loss_clip_max:
                logger.warning(f"Loss过大: {mae_loss.item():.2f},裁剪到50")
                mae_loss = torch.clamp(mae_loss, max=args.loss_clip_max)


            # ========== Backward ==========
            scaler.scale(mae_loss).backward()

            # ========== 梯度累积 ==========
            if (i + 1) % accumulation_steps == 0:
                # ========== 优化17: 检查梯度 ==========

                # 梯度裁剪
                scaler.unscale_(optim)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(CustomAuroraModel.parameters(), max_grad_norm)

                # 优化器更新
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            # ========== 监控指标（不需要梯度） ==========
            with torch.no_grad():
                monitor_surf_channel = args.surf_channels[monitor_surf_var]
                monitor_atmos_channel = args.atmos_channels[monitor_atmos_var]
                if args.train_roll_step==1:
                    mse_var_surf = F.mse_loss(prediction[1].surf_vars[monitor_surf_var][:, 0], target[:, monitor_surf_channel])
                    mse_var_atmos = F.mse_loss(prediction[1].atmos_vars[monitor_atmos_var][:, 0],target[:, monitor_atmos_channel[0]:monitor_atmos_channel[1]])
                else:
                    mse_var_surf = F.mse_loss(prediction.surf_vars[monitor_surf_var][:, 0], target[:, monitor_surf_channel])
                    mse_var_atmos = F.mse_loss(prediction.atmos_vars[monitor_atmos_var][:, 0],target[:, monitor_atmos_channel[0]:monitor_atmos_channel[1]])


            total_loss += mae_loss.item() * accumulation_steps
            train_iter.set_postfix({
                "MAE loss": f"{mae_loss.item() * accumulation_steps:.6f}",
                f"rmse_{monitor_surf_var}": f"{mse_var_surf.item()**0.5:.6f}",
                f"rmse_{monitor_atmos_var}": f"{mse_var_atmos.item()**0.5:.6f}",

            })

            # ========== 优化18: 及时清理显存 ==========
            del prediction, new_target, mae_loss, batch


        # epoch 末尾处理剩余累积梯度
        if len(train_loader) % accumulation_steps != 0:
            scaler.unscale_(optim)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(CustomAuroraModel.parameters(), max_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        # ========== Epoch结束 ==========
        avg_loss = distributed_mean(total_loss / len(train_loader), context)

        lr_scheduler.step()

        # ========== 优化19: 保存更多信息到checkpoint ==========
        if context.is_main and args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or (epoch + 1) == epochs):
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
        barrier(context)

        # 评估
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == epochs):
            barrier(context)
            if context.is_main:
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # 评估前清理显存
                evaluate(CustomAuroraModel, test_loader, args, logger, device=str(device))
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # 评估后清理显存
            barrier(context)

    logger.info("🎉 训练完成!")


if __name__ == "__main__":
    args = get_args()
    context = initialize_distributed(args)
    seed_everything(args.seed, context.rank)
    logger = setup_logging(args.log_file, args.log_dir, context.is_main, context.rank)
    try:
        if context.is_main:
            print(args)
        logger.info("🚀 -----------------开始训练---------------")
        logger.info(f"训练参数: {args}")
        logger.info(
            f"distributed={context.distributed}, world_size={context.world_size}, device={context.device}"
        )
        if context.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(context.device)}")
            logger.info(
                f"总显存: {torch.cuda.get_device_properties(context.device).total_memory / 1024**3:.2f} GB"
            )
        main(args, logger, context)
    finally:
        cleanup_distributed(context)

