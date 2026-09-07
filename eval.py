import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（也就是 qwe 文件夹的路径）
project_root = os.path.dirname(current_dir)
# 将 qwe 目录加入到系统搜索路径中
sys.path.append(project_root)
import torch
from tqdm import tqdm
import torch.nn.functional as F
from aurora import AuroraPretrained,AuroraSmallPretrained, Batch, Metadata,rollout
from utils import hours_to_datetime
from utils import static_var,calculate_batch_rmse_sum,hours_to_datetime,fix_tensor,build_batch_from_tensor
from data.weather_dataset import WeatherBench2, custom_collate
from torch.utils.data import Dataset, DataLoader
from model import CustomAurora
from aurora.normalisation import locations, scales
from args import get_args
import os
from datetime import timedelta

def compute_rmse(loss_dict,final_pred,target,args):
    for name in args.eval_vars:
        if name in final_pred.surf_vars and name in args.surf_channels:
            loss_dict[name] += calculate_batch_rmse_sum(
                final_pred.surf_vars[name][:, 0], target[:, args.surf_channels[name]]
            )
        elif name in final_pred.atmos_vars and name in args.atmos_channels:
            start, end = args.atmos_channels[name]
            loss_dict[name] += calculate_batch_rmse_sum(final_pred.atmos_vars[name][:, 0], target[:, start:end])
        else:
            raise KeyError(f"评估变量 {name} 不存在于 prediction 或 channel 配置中")
    return loss_dict


def _log_message(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def _has_invalid_tensor(*tensors):
    return any(torch.isnan(tensor).any() or torch.isinf(tensor).any() for tensor in tensors)


def _fix_invalid_tensors(*tensors):
    return tuple(fix_tensor(tensor) for tensor in tensors)


def _stack_rollout_targets(targets, device):
    if 'rollout_tgts' not in targets[0]:
        raise KeyError("多步 rollout 评估需要 WeatherBench2 返回 rollout_tgts")
    return [
        torch.stack([target['rollout_tgts'][lead_idx] for target in targets], dim=0).to(device)
        for lead_idx in range(len(targets[0]['rollout_tgts']))
    ]


def _validate_lead_count(predictions, rollout_targets, args):
    if len(predictions) != args.roll_step or len(rollout_targets) != args.roll_step:
        raise ValueError(
            f"rollout 预测步数和目标步数不一致: "
            f"pred={len(predictions)}, target={len(rollout_targets)}, args.roll_step={args.roll_step}"
        )


def _summarize_lead_rmse(lead_loss_dicts, total_samples, args):
    if total_samples == 0:
        raise ValueError("评估集为空，无法计算 RMSE")
    lead_results = []
    for lead_idx, loss_dict in enumerate(lead_loss_dicts, start=1):
        avg_rmse_dict = {name: value / total_samples for name, value in loss_dict.items()}
        avg_total_rmse = sum(avg_rmse_dict.values()) / len(avg_rmse_dict)
        lead_results.append({
            "lead_step": lead_idx,
            "lead_days": lead_idx * args.timestep / 24,
            "avg": avg_total_rmse,
            "vars": avg_rmse_dict,
        })
    return lead_results


def _print_lead_rmse_results(lead_results, logger):
    for result in lead_results:
        lead_idx = result["lead_step"]
        lead_days = result["lead_days"]
        _log_message(logger, f"\n📊 Aurora weighted lead {lead_idx:02d} ({lead_days:.2f} day) RMSE Results:")
        _log_message(logger, f"  - RMSE [ avg]: {result['avg']:.12f}")
        for var, rmse_val in result["vars"].items():
            _log_message(logger, f"  - RMSE [{var:>4}]: {rmse_val:.12f}")

@torch.no_grad()
def evaluate(model, test_loader,args, logger,device="cuda", return_details=False):
    static_vars_z, static_vars_lsm, static_vars_slt = static_var(
        static_path=args.static_path,
        target_size=(args.target_height, args.target_width),
    )
    args.static_vars_z = static_vars_z.to(device).requires_grad_(False)
    args.static_vars_lsm = static_vars_lsm.to(device).requires_grad_(False)
    args.static_vars_slt = static_vars_slt.to(device).requires_grad_(False)
    model.eval()

    # 初始化各变量的 loss 累加器
    lead_loss_dicts = [
        {name: 0.0 for name in args.eval_vars}
        for _ in range(args.roll_step)
    ]

    total_samples = 0 # 记录总的时间步数 (T)
    reused_samples = 0
    with torch.no_grad():
        for (images, targets) in tqdm(test_loader, desc="Evaluating", leave=True):
            save_path = None
            if args.roll_step > 1:
                rollout_targets = _stack_rollout_targets(targets, device)
                if args.save_pt:
                    os.makedirs(args.save_pt_dir, exist_ok=True)
                    save_path = os.path.join(args.save_pt_dir, f"{targets[0]['filename']}.pt")
                    if os.path.isfile(save_path):
                        if _has_invalid_tensor(*rollout_targets):
                            rollout_targets = list(_fix_invalid_tensors(*rollout_targets))
                        prediction = torch.load(save_path, map_location="cpu", weights_only=False)
                        _validate_lead_count(prediction, rollout_targets, args)
                        batch_size = rollout_targets[0].shape[0]
                        total_samples += batch_size
                        reused_samples += batch_size
                        for lead_idx, pred in enumerate(prediction):
                            lead_loss_dicts[lead_idx] = compute_rmse(
                                lead_loss_dicts[lead_idx], pred.to(device), rollout_targets[lead_idx], args
                            )
                        continue
            else:
                rollout_targets = [torch.stack([t['tgt'] for t in targets], dim=0).to(device)]

            images_1 = torch.stack([im[0].float() for im in images], dim=0).to(device)
            images_2 = torch.stack([im[1] for im in images], dim=0).to(device)
            if args.roll_step > 1:
                tensors_to_check = (images_1, images_2, *rollout_targets)
                if _has_invalid_tensor(*tensors_to_check):
                    fixed_tensors = _fix_invalid_tensors(*tensors_to_check)
                    images_1, images_2 = fixed_tensors[0], fixed_tensors[1]
                    rollout_targets = list(fixed_tensors[2:])
            else:
                if _has_invalid_tensor(images_1, images_2, rollout_targets[0]):
                    images_1, images_2, rollout_targets[0] = _fix_invalid_tensors(images_1, images_2, rollout_targets[0])
            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            args.lat_tensor = torch.linspace(90, -90, args.target_height + 1)[:-1].to(device)
            args.lon_tensor = torch.linspace(0, 360, args.target_width + 1)[:-1].to(device)
            args.atmos_levels = tuple(args.atmos_levels)
            batch=build_batch_from_tensor(images_2, time, args, history_tensor=images_1)


            batch_size = rollout_targets[0].shape[0]
            total_samples += batch_size # 累加 T
            with torch.inference_mode():
                if args.roll_step > 1:
                    prediction = [pred.to("cpu") for pred in rollout(model, batch, steps=args.roll_step,args=args)]
                    _validate_lead_count(prediction, rollout_targets, args)
                    if args.save_pt:
                        torch.save(prediction, save_path)
                    for lead_idx, pred in enumerate(prediction):
                        lead_loss_dicts[lead_idx] = compute_rmse(
                            lead_loss_dicts[lead_idx], pred.to(device), rollout_targets[lead_idx], args
                        )
                else:
                    prediction = model(batch, args)
                    final_pred = prediction[1]
                    lead_loss_dicts[0]=compute_rmse(lead_loss_dicts[0],final_pred,rollout_targets[0],args)

                # --- 核心修改：加权 RMSE 而不是 MSE ---
             # 结果计算：除以总样本数 T
    if reused_samples:
        _log_message(logger, f"Reused {reused_samples} existing prediction file(s); model rollout was skipped.")
    lead_results = _summarize_lead_rmse(lead_loss_dicts, total_samples, args)
    _print_lead_rmse_results(lead_results, logger)
    if return_details:
        return lead_results
    return lead_results[-1]["avg"]


if __name__ == "__main__":
    args = get_args()
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = AuroraPretrained(
        autocast=args.use_amp,
        use_lora=args.use_lora,
        stabilise_level_agg=args.stabilise_level_agg,
        timestep=timedelta(hours=args.timestep),
        surf_vars=tuple(args.surf_vars),
        atmos_vars=tuple(args.atmos_vars),
        # surf_stats=surf_stats
    )


    model.load_checkpoint_local(args.pretrained,strict=False)
    model=model.to(device)
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

    # ========== 优化5: 降低学习率，增加稳定性 ==========
    param_dicts = [
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and ('lora_proj' in n or
                                             'lora_qkv' in n or 'hypernetwork' in n)],
            "lr": args.lr1,  # 1e-4
        },
        {
            "params": [p for n, p in CustomAuroraModel.named_parameters()
                      if p.requires_grad and (n.startswith('model.encoder.surf_token_embeds.weights') or
                                             n.startswith('model.encoder.atmos_token_embeds.weights') or
                                             ('surf_heads' in n and 'hypernetwork' not in n) or ('atmos_heads' in n and 'hypernetwork' not in n))],
            "lr": args.lr2,  # 5e-5
        },
    ]

    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters() if p.requires_grad)
    print('number of leanable params:', n_parameters)

    n_parameters = sum(p.numel() for p in CustomAuroraModel.parameters())
    print('number of all params:', n_parameters)
    optim = torch.optim.AdamW(
        param_dicts, lr=3e-4,
        weight_decay=1e-4
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10)



    if not args.ckpt:
        raise ValueError("独立运行 eval.py 时必须通过 --ckpt 指定待评估 checkpoint")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    current_epoch = checkpoint['epoch']
    print(f"🔁 加载最新断点: {args.ckpt}  epoch: {current_epoch}")
    CustomAuroraModel.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # print(f"✅ 成功恢复到 epoch {start_epoch}")

    CustomAuroraModel.eval()

    test_dataset= WeatherBench2(
        data_folder=args.data_folder,
        roll_step=(args.roll_step-1),
        timestep=args.timestep,
        years=args.test_year,
        args=args,
        num_rollout_targets=args.roll_step if args.roll_step > 1 else 0,
        return_train_aux_target=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.test_num_workers, pin_memory=args.test_pin_memory, drop_last=args.drop_last)
    # 假设 test_loader 已经定义并加载了测试数据
    # 例如: test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # evaluate 函数会打印每个变量的 MSE loss 和平均总 loss
    logger=None
    evaluate(CustomAuroraModel, test_loader,args,logger,device=str(device))  # 请确保 test_loader 已定义并包含测试数据
