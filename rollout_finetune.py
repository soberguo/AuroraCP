import torch
import torch.nn.functional as F
from aurora import AuroraPretrained
from torch.amp import GradScaler
from tqdm import tqdm
from utils import compute_loss, hours_to_datetime, static_var, construct_batch, build_batch_from_tensor
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
    distributed_any_true,
    distributed_mean,
    initialize_distributed,
    seed_everything,
    wrap_ddp,
)
import logging
import sys
from datetime import timedelta
from rollout_finetune.buffer import AuroraReplayBuffer
from rollout_finetune.checkpointing import (
    STAGE3_CHECKPOINT_VERSION,
    STAGE3_NAME,
    atomic_torch_save,
    capture_rng_state,
    load_rank_state,
    prune_rank_states,
    restore_rng_state,
    save_rank_state,
    wait_for_file,
)

# ==========================================
# 1. 辅助函数：Tensor 与 Aurora Batch 的转换
# ==========================================
def _put_var_to_tensor(output, value, channel):
    if isinstance(channel, int):
        output[:, channel] = value[:, 0] if value.ndim == 4 else value
        return
    start, end = channel
    if value.ndim == 5:
        value = value.squeeze(1)
    output[:, start:end] = value


def stack_vars_to_tensor(pred_denorm, reference_tensor, args):
    """
    将 Aurora 输出还原到 dataset 使用的 (B, C, H, W) channel 布局。
    """
    output = reference_tensor.clone()
    for name in args.surf_vars:
        _put_var_to_tensor(output, pred_denorm.surf_vars[name], args.surf_channels[name])
    for name in args.atmos_vars:
        _put_var_to_tensor(output, pred_denorm.atmos_vars[name], args.atmos_channels[name])
    return output


def evaluate_on_main_rank(model, test_loader, args, logger, device, context, epoch):
    """Run rank-0-only evaluation without leaving an NCCL collective pending."""
    if not context.distributed:
        evaluate(model, test_loader, args, logger, device=str(device))
        return

    marker_path = os.path.join(args.save_dir, 'rank_states', f'eval_epoch_{epoch:03d}.done')
    if context.is_main and os.path.exists(marker_path):
        os.remove(marker_path)
    barrier(context)

    if context.is_main:
        rng_state = capture_rng_state(device)
        try:
            evaluate(model, test_loader, args, logger, device=str(device))
        finally:
            restore_rng_state(rng_state, device)
        atomic_torch_save({'epoch': epoch}, marker_path)
    else:
        wait_for_file(marker_path)

    barrier(context)
    if context.is_main:
        os.remove(marker_path)

# ==========================================
# 2. 日志设置
# ==========================================
def setup_logging(log_file="training_log.txt", log_dir="log", is_main=True, rank=0):
    """
    设置日志，保存到指定文件夹

    Args:
        log_file: 日志文件名
        log_dir: 日志文件夹路径
    """
    logger = logging.getLogger(f"aurora.rollout.rank{rank}")
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
# ==========================================
# 3. 主函数
# ==========================================
def main(args, logger, context):
    device = context.device
    logger.info(f"Using device: {device}")

    # --- Dataset & DataLoader ---
    # 注意：train=True 时 dataset 会加载大量数据
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

    # --- Static Vars ---
    static_vars_z, static_vars_lsm, static_vars_slt = static_var(
        static_path=args.static_path,
        target_size=(args.target_height, args.target_width),
    )
    args.static_vars_z = static_vars_z.to(device).requires_grad_(False)
    args.static_vars_lsm = static_vars_lsm.to(device).requires_grad_(False)
    args.static_vars_slt = static_vars_slt.to(device).requires_grad_(False)

    # --- Model Init ---
    model = AuroraPretrained(
        autocast=args.use_amp,
        use_lora=args.use_lora,
        stabilise_level_agg=args.stabilise_level_agg,
        timestep=timedelta(hours=args.timestep),
        surf_vars=tuple(args.surf_vars),
        atmos_vars=tuple(args.atmos_vars),
    )
    # 加载预训练权重
    if os.path.exists(args.pretrained):
        model.load_checkpoint_local(args.pretrained, strict=False)
    else:
        logger.warning(f"未找到预训练权重 {args.pretrained}，将使用随机初始化！")

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



    # --- Replay Buffer ---
    # 单卡 Buffer Size (论文总共4000，单卡200，这里我们单卡设为400以增加多样性)
    replay_buffer = AuroraReplayBuffer(capacity=args.replay_buffer_capacity)
    dataset_sampling_period = args.dataset_sampling_period

    # --- Tensors ---
    args.lat_tensor = torch.linspace(90, -90, args.target_height + 1)[:-1].to(device)
    args.lon_tensor = torch.linspace(0, 360, args.target_width + 1)[:-1].to(device)
    args.atmos_levels = tuple(args.atmos_levels)

    max_grad_norm = args.max_grad_norm
    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    # --- Training Loop ---
    start_epoch = 0
    epochs = args.epochs
    global_step = 0

    for n, p in CustomAuroraModel.named_parameters():
        # if "norm" in n:
        #     p.requires_grad = True
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

    # ========== 优化5: 降低学习率，增加稳定性 ==========
    param_dicts = [
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (('lora_proj' in n and 'hypernetwork' not in n) or
                                             ('lora_qkv' in n and 'hypernetwork' not in n))],
            "lr": args.lr1,  # 1e-4
        },
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (n.startswith('model.encoder.surf_token_embeds.weights') or
                                             n.startswith('model.encoder.atmos_token_embeds.weights') or
                                             'surf_heads' in n or 'atmos_heads' in n or 'hypernetwork' in n)],
            "lr": args.lr2,  # 5e-5
        },
    ]

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
    save_dir = args.save_dir
    if context.is_main:
        os.makedirs(save_dir, exist_ok=True)
    barrier(context)
    # Resume训练：传入 --ckpt 即恢复，不传则从 pretrained 初始化
    resume_rng_state = None
    if args.ckpt:
        resume_path= args.ckpt
        logger.info(f"🔁 加载最新断点: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        start_epoch = checkpoint['epoch']
        is_stage3_checkpoint = (
            checkpoint.get('training_stage') == STAGE3_NAME
            or 'global_step' in checkpoint
        )
        CustomAuroraModel.load_state_dict(
            checkpoint['model_state_dict'],
            strict=is_stage3_checkpoint,
        )
        if is_stage3_checkpoint:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint['global_step']
            logger.info(
                "已恢复 Stage 3 optimizer/scheduler: "
                f"lr={[group['lr'] for group in optim.param_groups]}, "
                f"scheduler_last_epoch={lr_scheduler.last_epoch}, global_step={global_step}"
            )

            checkpoint_version = checkpoint.get('checkpoint_format_version', 1)
            if checkpoint_version >= STAGE3_CHECKPOINT_VERSION:
                saved_world_size = checkpoint.get('world_size')
                if saved_world_size != context.world_size:
                    raise ValueError(
                        "严格恢复要求 world_size 不变: "
                        f"checkpoint={saved_world_size}, current={context.world_size}"
                    )
                resume_rng_state, rank_state_file = load_rank_state(
                    save_dir=os.path.dirname(resume_path),
                    epoch=start_epoch,
                    rank=context.rank,
                    world_size=context.world_size,
                    replay_buffer=replay_buffer,
                )
                logger.info(
                    f"已恢复 rank {context.rank} replay/RNG state: "
                    f"{rank_state_file} ({len(replay_buffer)} items)"
                )
            else:
                logger.warning(
                    "该 Stage 3 checkpoint 是旧格式，不含 replay buffer/RNG；"
                    "optimizer、scheduler、AMP scaler 将连续，但本次恢复无法做到严格数值连续。"
                )
        else:
            logger.info("从 Stage 2 初始化 Stage 3，使用新的 optimizer、scheduler 和 replay buffer")
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"✅ 成功恢复到 epoch {start_epoch}")

    training_model = wrap_ddp(
        CustomAuroraModel,
        context,
        find_unused_parameters=args.ddp_find_unused_parameters,
    )
    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state, device)

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        total_loss = 0.0

        # 转换 DataLoader 为 Iterator
        train_iterator = iter(train_loader)
        steps_per_epoch = len(train_loader)

        # === 动态 Rollout 策略 (User Requested) ===
        # 前 5 epoch: 4天 (16步，假设6h数据)
        # 后 5 epoch: 10天 (40步)
        if epoch < args.rollout_curriculum_switch_epoch:
            current_max_lead = args.rollout_short_max_lead
        else:
            current_max_lead = args.rollout_long_max_lead

        logger.info(f"Epoch {epoch+1}: Max Lead Time set to {current_max_lead} steps")

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not context.is_main,
        )

        for _ in pbar:
            global_step += 1

            # -----------------------------------------------------------
            # 1. Fresh Injection (注入新鲜数据)
            # -----------------------------------------------------------
            # 每隔 N 步，或者 Buffer 为空时，从 DataLoader 读取数据
            needs_fresh_data = global_step % dataset_sampling_period == 0 or len(replay_buffer) < args.batchsize
            if distributed_any_true(needs_fresh_data, context):
                try:
                    fresh_batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    fresh_batch = next(train_iterator)

                images, targets = fresh_batch

                img_prev = torch.stack([im[0].float() for im in images], dim=0) #(B, C, H, W)
                img_curr = torch.stack([im[1].float() for im in images], dim=0)
                local_fresh_valid = not (
                    torch.isnan(img_prev).any().item()
                    or torch.isinf(img_prev).any().item()
                    or torch.isnan(img_curr).any().item()
                    or torch.isinf(img_curr).any().item()
                )
                if not distributed_all_true(local_fresh_valid, context):
                    continue  # 跳过这个 batch，防止训练崩溃
                    # images[b] = (x_prev, x_curr)
                # 获取 T 时刻的文件名 (metadata)
                filenames = [t['filename'] for t in targets]

                # 初始 lead_time = 0
                lead_times = torch.zeros(img_prev.shape[0], dtype=torch.long)

                # 推入 Buffer
                replay_buffer.push(img_prev, img_curr, filenames, lead_times)

            # 冷启动期间，如果 Buffer 不够，先跳过训练
            if not distributed_all_true(len(replay_buffer) >= args.batchsize, context):
                continue

            # -----------------------------------------------------------
            # 2. Sample from Buffer (采样)
            # -----------------------------------------------------------
            batch_data = replay_buffer.sample(args.batchsize)

            b_prev = batch_data['prev'].to(device) # T-1
            b_curr = batch_data['curr'].to(device) # T
            b_fnames = batch_data['fname']         # Filename of T
            b_leads = batch_data['lead'].to(device)

            # -----------------------------------------------------------
            # 3. Dynamic Target Loading (加载真值)
            # -----------------------------------------------------------
            # 根据 T 时刻的文件名，去磁盘找 T+1 的真值
            gt_list = []
            valid_mask = []

            for fname in b_fnames:
                tgt_tensor, next_fname = dataset.get_next_target(fname)
                if tgt_tensor is not None:
                    gt_list.append(tgt_tensor.float().to(device))
                    valid_mask.append(True)
                else:
                    # 找不到 target (可能是数据集末尾)，给个占位符
                    gt_list.append(torch.zeros_like(b_curr[0]))
                    valid_mask.append(False)

            if not distributed_all_true(any(valid_mask), context):
                continue
            time_objs = tuple(hours_to_datetime(f) for f in b_fnames)
            b_target = torch.stack(gt_list, dim=0) # T+1
            local_target_valid = not (
                torch.isnan(b_target).any().item() or torch.isinf(b_target).any().item()
            )
            if not distributed_all_true(local_target_valid, context):
                continue  # 跳过这个 batch，防止训练崩溃
            with torch.no_grad():
                target_batch=construct_batch(b_target,time_objs,args)

                new_target = CustomAuroraModel.biaozhunhua(target_batch)
                del target_batch
            # -----------------------------------------------------------
            # 4. Forward & Backward (Pushforward Trick)
            # -----------------------------------------------------------


            # 构建 Aurora Batch
            input_batch = build_batch_from_tensor(b_curr, time_objs, args, history_tensor=b_prev)


            prediction = training_model(input_batch, args)
            mae_loss=compute_loss(prediction[0], new_target, args)

            pred_tensor = stack_vars_to_tensor(prediction[1], b_curr, args)

            if args.loss_clip_max > 0 and mae_loss > args.loss_clip_max:
                logger.warning(f"Loss过大: {mae_loss.item():.2f},裁剪到{args.loss_clip_max}")
                mae_loss = torch.clamp(mae_loss, max=args.loss_clip_max)

            scaler.scale(mae_loss).backward()

            if global_step % accumulation_steps == 0:
                scaler.unscale_(optim)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(CustomAuroraModel.parameters(), max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            total_loss += mae_loss.item()
            # pbar.set_postfix(loss=mae_loss.item(), buf=len(replay_buffer))
            with torch.no_grad():
                monitor_surf_var = args.main_surf_vars[0] if args.main_surf_vars else args.surf_vars[0]
                monitor_atmos_var = args.atmos_vars[0]
                monitor_surf_channel = args.surf_channels[monitor_surf_var]
                monitor_atmos_channel = args.atmos_channels[monitor_atmos_var]
                mse_var_surf = F.mse_loss(prediction[1].surf_vars[monitor_surf_var][:, 0], b_target[:, monitor_surf_channel])
                mse_var_atmos = F.mse_loss(prediction[1].atmos_vars[monitor_atmos_var][:, 0], b_target[:, monitor_atmos_channel[0]:monitor_atmos_channel[1]])

            # total_loss += mae_loss.item() * accumulation_steps
            pbar.set_postfix({
                "buffer_size": len(replay_buffer),
                "MAE loss": f"{mae_loss.item() * accumulation_steps:.6f}",
                f"rmse_{monitor_surf_var}": f"{mse_var_surf.item()**0.5:.6f}",
                f"rmse_{monitor_atmos_var}": f"{mse_var_atmos.item()**0.5:.6f}",

            })
            # -----------------------------------------------------------
            # 5. Push Back to Buffer (回存)
            # -----------------------------------------------------------
            with torch.no_grad():
                next_prevs = []
                next_currs = []
                next_fnames = []
                next_leads = []

                for k in range(args.batchsize):
                    # 过滤条件:
                    # 1. 必须是 Valid (找到了 Target)
                    # 2. Lead Time 未超过当前阶段限制
                    if not valid_mask[k]: continue
                    if b_leads[k] >= current_max_lead: continue

                    # 查找 Next Filename (用于下一次迭代找 Target)
                    # b_fnames[k] 是 T，我们需要 T+1 的文件名
                    _, next_fname = dataset.get_next_target(b_fnames[k])
                    if next_fname is None: continue

                    # Push: (T, Pred_T+1) -> 预测 T+2
                    next_prevs.append(b_curr[k])        # 原来的 Curr (T) 变成 Prev
                    next_currs.append(pred_tensor[k])   # 预测值 (T+1) 变成 Curr
                    next_fnames.append(next_fname)      # T+1 的文件名
                    next_leads.append(b_leads[k] + 1)

                if len(next_prevs) > 0:
                    replay_buffer.push(
                        torch.stack(next_prevs),
                        torch.stack(next_currs),
                        next_fnames,
                        torch.stack(next_leads)
                    )

        # --- End of Epoch ---
        avg_loss = distributed_mean(total_loss / steps_per_epoch, context)
        lr_scheduler.step()

        # Save the per-rank replay/RNG state before publishing the main checkpoint.
        should_save = args.save_every > 0 and (
            (epoch + 1) % args.save_every == 0 or (epoch + 1) == epochs
        )
        if should_save:
            saved_epoch = epoch + 1
            rank_state_file = save_rank_state(
                save_dir=args.save_dir,
                epoch=saved_epoch,
                rank=context.rank,
                world_size=context.world_size,
                replay_buffer=replay_buffer,
                device=device,
            )
            logger.info(f"Saved rank state to {rank_state_file}")
            barrier(context)

        if context.is_main and should_save:
            save_path = os.path.join(args.save_dir, f"epoch_{saved_epoch:03d}.pt")
            atomic_torch_save({
                'checkpoint_format_version': STAGE3_CHECKPOINT_VERSION,
                'training_stage': STAGE3_NAME,
                'world_size': context.world_size,
                'epoch': saved_epoch,
                'model_state_dict': CustomAuroraModel.state_dict(), # 无 .module
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), # 保存 scaler
                'avg_loss': avg_loss,
                'global_step': global_step,
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
        barrier(context)

        if should_save:
            prune_rank_states(args.save_dir, context.rank, args.replay_state_keep)

        # Eval

        should_evaluate = (
            args.eval_every > 0
            and (epoch + 1) >= args.eval_start_epoch
            and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == epochs)
        )
        if should_evaluate:
            evaluate_on_main_rank(
                CustomAuroraModel,
                test_loader,
                args,
                logger,
                device,
                context,
                epoch + 1,
            )


    logger.info("Training Finished!")

if __name__ == "__main__":
    args = get_args()
    context = initialize_distributed(args)
    seed_everything(args.seed, context.rank)
    logger = setup_logging(args.log_file, args.log_dir, context.is_main, context.rank)
    try:
        logger.info(str(args))
        logger.info(
            f"distributed={context.distributed}, world_size={context.world_size}, device={context.device}"
        )
        if context.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(context.device)}")
        main(args, logger, context)
    finally:
        cleanup_distributed(context)
