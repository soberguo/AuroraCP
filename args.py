import argparse


DEFAULT_MAIN_SURF_VARS = ["2t", "10u", "10v", "msl"]
DEFAULT_LAND_VARS = ["sshf", "slhf", "vswl"]
DEFAULT_ATMOS_VARS = ["z", "u", "v", "t", "q"]
DEFAULT_EVAL_VARS = DEFAULT_MAIN_SURF_VARS  + DEFAULT_ATMOS_VARS
DEFAULT_ATMOS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

DEFAULT_SURF_LOSS_WEIGHTS = [3.5, 0.77, 0.66, 1.6]
DEFAULT_ATMOS_LOSS_WEIGHTS = [3.5, 0.87, 0.6, 1.7, 0.8]

DEFAULT_MAIN_SURF_CHANNELS = {
    "2t": 0,
    "10u": 1,
    "10v": 2,
    "msl": 3,
}
DEFAULT_LAND_CHANNELS = {
    "sshf": 69,
    "slhf": 70,
    "vswl": 71,
}
DEFAULT_ATMOS_CHANNELS = {
    "z": [4, 17],
    "u": [17, 30],
    "v": [30, 43],
    "t": [43, 56],
    "q": [56, 69],
}


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _finalize_variable_config(args):
    if args.dataset_sampling_period <= 0:
        raise ValueError("--dataset_sampling_period 必须大于 0")
    if args.replay_buffer_capacity < args.batchsize:
        raise ValueError("--replay_buffer_capacity 必须大于等于 --batchsize")
    if args.replay_state_keep <= 0:
        raise ValueError("--replay_state_keep 必须大于 0")
    if args.eval_start_epoch < 0:
        raise ValueError("--eval_start_epoch 必须大于等于 0")
    if args.rollout_short_max_lead < 0 or args.rollout_long_max_lead < 0:
        raise ValueError("rollout max lead 参数必须大于等于 0")
    # args.surf_vars = args.main_surf_vars + args.land_vars
    # args.surf_channels = {**args.main_surf_channels, **args.land_channels}
    args.surf_vars = list(args.main_surf_vars)
    args.surf_channels = dict(args.main_surf_channels)
    if args.eval_vars is None:
        args.eval_vars = args.surf_vars + args.atmos_vars

    if len(args.surf_loss_weights) != len(args.surf_vars):
        raise ValueError("--surf_loss_weights 的数量必须和 main_surf_vars + land_vars 一致")
    if len(args.atmos_loss_weights) != len(args.atmos_vars):
        raise ValueError("--atmos_loss_weights 的数量必须和 --atmos_vars 一致")
    duplicated_surf_vars = {name for name in args.surf_vars if args.surf_vars.count(name) > 1}
    if duplicated_surf_vars:
        raise ValueError(f"main_surf_vars 和 land_vars 中存在重复变量: {sorted(duplicated_surf_vars)}")
    missing_main_channels = set(args.main_surf_vars) - set(args.main_surf_channels)
    if missing_main_channels:
        raise ValueError(f"main_surf_vars 缺少 channel 配置: {sorted(missing_main_channels)}")
    missing_land_channels = set(args.land_vars) - set(args.land_channels)
    if missing_land_channels:
        raise ValueError(f"land_vars 缺少 channel 配置: {sorted(missing_land_channels)}")
    missing_atmos_channels = set(args.atmos_vars) - set(args.atmos_channels)
    if missing_atmos_channels:
        raise ValueError(f"atmos_vars 缺少 channel 配置: {sorted(missing_atmos_channels)}")
    valid_eval_vars = set(args.surf_channels) | set(args.atmos_channels)
    unknown_eval_vars = set(args.eval_vars) - valid_eval_vars
    if unknown_eval_vars:
        raise ValueError(f"eval_vars 中存在未知变量或缺少 channel 配置: {sorted(unknown_eval_vars)}")
    return args

def get_args():
    parser = argparse.ArgumentParser()

    # 基础训练参数
    parser.add_argument('--ckpt', default=None, type=str,
                        help='训练恢复/评估 checkpoint 路径；不传则从 pretrained 初始化训练')
    parser.add_argument('--save_dir', default='original', type=str)
    parser.add_argument('--device', default='cuda', type=str,
                        help='训练设备，如 cuda、cuda:0 或 cpu')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')
    parser.add_argument('--distributed', default=False, type=_str_to_bool,
                        help='是否由 torchrun 启动 DDP；单卡普通 python 启动时设为 false')
    parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'gloo'],
                        help='PyTorch distributed backend')
    parser.add_argument('--ddp_find_unused_parameters', default=True, type=_str_to_bool,
                        help='是否让 DDP 检测未参与当前 forward 的可训练参数')
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--train_roll_step', default=4, type=int)
    parser.add_argument('--replay_buffer_capacity', default=400, type=int,
                        help='rollout_finetune replay buffer 容量')
    parser.add_argument('--replay_state_keep', default=2, type=int,
                        help='每个 rank 保留最近多少个 replay/RNG sidecar')
    parser.add_argument('--dataset_sampling_period', default=10, type=int,
                        help='rollout_finetune 每多少个 global step 注入一次真实数据')
    parser.add_argument('--rollout_curriculum_switch_epoch', default=50, type=int,
                        help='rollout_finetune 从短 lead 切换到长 lead 的 epoch')
    parser.add_argument('--rollout_short_max_lead', default=12, type=int,
                        help='rollout_finetune curriculum 前期最大 lead step')
    parser.add_argument('--rollout_long_max_lead', default=20, type=int,
                        help='rollout_finetune curriculum 后期最大 lead step')
    parser.add_argument('--save_every', default=1, type=int,
                        help='每多少个 epoch 保存一次 checkpoint')
    parser.add_argument('--eval_every', default=1, type=int,
                        help='每多少个 epoch 评估一次；0 表示不在训练中评估')
    parser.add_argument('--eval_start_epoch', default=0, type=int,
                        help='从哪个全局 epoch 开始训练内评估（包含该 epoch）')
    parser.add_argument('--lr1', default=1e-3, type=float)
    parser.add_argument('--lr2', default=1e-4, type=float)
    parser.add_argument('--base_lr', default=1e-4, type=float,
                        help='AdamW 默认学习率；参数组未指定 lr 时使用')
    parser.add_argument('--reset_lr_on_resume', default=False, type=_str_to_bool,
                        help='加载 checkpoint 后使用命令行学习率并重新开始 scheduler；用于阶段切换')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='AdamW weight decay')
    parser.add_argument('--adam_eps', default=1e-8, type=float,
                        help='AdamW eps')
    parser.add_argument('--adam_betas', default=[0.9, 0.999], type=float, nargs=2,
                        help='AdamW beta1 beta2')
    parser.add_argument('--scheduler_step_size', default=10, type=int,
                        help='StepLR step_size')
    parser.add_argument('--scheduler_gamma', default=0.5, type=float,
                        help='StepLR gamma')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='梯度裁剪阈值；<=0 表示不裁剪')
    parser.add_argument('--loss_clip_max', default=50.0, type=float,
                        help='loss 最大裁剪值；<=0 表示不裁剪')
    parser.add_argument('--train_year', default=[2013, 2018], type=int, nargs=2,
                        help='训练年份范围，如: --train_year 1979 2018')
    parser.add_argument('--test_year', default=[2019,2020], type=int, nargs=2,
                        help='测试年份范围，如: --test_year 2020 2021')

    # 时间步长参数: 6 或 12 小时
    parser.add_argument('--timestep', default=6, type=int, choices=[6, 12],
                        help='预测时间步长：6小时或12小时')
    parser.add_argument('--pretrained', default='ckpt/aurora-0.25-pretrained.ckpt', type=str,
                        help='pretrained or finetuned')
    # 数据路径
    parser.add_argument('--data_folder', default='/sharefiles1/guoyixin/datasets/weatherbench2_73var', type=str)
    parser.add_argument('--static_path', default='ckpt/aurora-0.25-static.pickle', type=str,
                        help='Aurora 静态变量 pickle 路径')
    parser.add_argument('--target_height', default=120, type=int,
                        help='训练/评估网格高度')
    parser.add_argument('--target_width', default=240, type=int,
                        help='训练/评估网格宽度')

    # 日志文件名
    parser.add_argument('--log_file', default='training_log.txt', type=str)
    parser.add_argument('--log_dir', default='log', type=str)

    # 梯度累积
    parser.add_argument('--accumulation_steps', default=1, type=int)

    # 模型开关
    parser.add_argument('--use_lora', default=True, type=_str_to_bool,
                        help='是否启用 Aurora backbone LoRA')
    parser.add_argument('--use_amp', default=True, type=_str_to_bool,
                        help='是否启用 autocast/AMP')
    parser.add_argument('--activation_checkpointing', default=True, type=_str_to_bool,
                        help='是否启用 activation checkpointing')
    parser.add_argument('--stabilise_level_agg', default=False, type=_str_to_bool,
                        help='Aurora stabilise_level_agg 开关')
    parser.add_argument('--enable_land_branch', default=True, type=_str_to_bool,
                        help='是否启用 CustomAurora.small_model 生成 hyp_x')

    # 集中变量配置
    parser.add_argument('--main_surf_vars', default=DEFAULT_MAIN_SURF_VARS, nargs='+',
                        help='直接送入 Aurora 主分支的 surface 变量')
    parser.add_argument('--land_vars', default=DEFAULT_LAND_VARS, nargs='+',
                        help='送入 Land Transformer / hypernetwork 的 surface 变量')
    parser.add_argument('--atmos_vars', default=DEFAULT_ATMOS_VARS, nargs='+',
                        help='Atmospheric 变量顺序')
    parser.add_argument('--eval_vars', default=None, nargs='+',
                        help='评估输出的变量列表；默认 main_surf_vars + land_vars + atmos_vars')
    parser.add_argument('--atmos_levels', default=DEFAULT_ATMOS_LEVELS, type=int, nargs='+',
                        help='气压层列表')
    parser.add_argument('--surf_loss_weights', default=DEFAULT_SURF_LOSS_WEIGHTS, type=float, nargs='+',
                        help='surface loss 权重，顺序与 main_surf_vars + land_vars 对齐')
    parser.add_argument('--atmos_loss_weights', default=DEFAULT_ATMOS_LOSS_WEIGHTS, type=float, nargs='+',
                        help='atmos loss 权重，顺序与 --atmos_vars 对齐')
    parser.add_argument('--surf_loss_scale', default=0.25, type=float,
                        help='surface loss 总体缩放系数')
    parser.add_argument('--loss_type', default='weighted_mae', type=str,
                        choices=['weighted_mae', 'mse'],
                        help='训练损失类型')
    parser.add_argument('--mask_ocean_for_land_vars', default=False, type=_str_to_bool,
                        help='是否对 land_vars 仅使用 lsm >= 0.5 的陆地像素计算训练损失')
    parser.set_defaults(main_surf_channels=DEFAULT_MAIN_SURF_CHANNELS)
    parser.set_defaults(land_channels=DEFAULT_LAND_CHANNELS)
    parser.set_defaults(atmos_channels=DEFAULT_ATMOS_CHANNELS)

    # FISH 参数
    parser.add_argument('--fish_keep_ratio', default=0.001, type=float,
                        help='FISH保留梯度比例(0,1]，越小越省显存和时间')
    parser.add_argument('--fish_grad_type', default='absolute', type=str, choices=['absolute', 'square'],
                        help='FISH重要性计算方式')
    parser.add_argument('--fish_noise_base', default=0.2, type=float,
                        help='FISH噪声权重基值，将按训练进度线性衰减')
    parser.add_argument('--fish_mask_interval', default=4, type=int,
                        help='每多少个optimizer step更新一次FISH mask，越大越快')
    parser.add_argument('--fish_lr', default=1e-3, type=float,
                        help='FISH backbone 微调学习率')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='mandatory 参数 weight decay')
    parser.add_argument('--fish_wd', default=1e-4, type=float,
                        help='FISH backbone weight decay（预训练权重建议0）')

    # DataLoader 参数
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test_num_workers', default=2, type=int)
    parser.add_argument('--pin_memory', default=True, type=_str_to_bool)
    parser.add_argument('--test_pin_memory', default=True, type=_str_to_bool)
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--persistent_workers', default=True, type=_str_to_bool)
    parser.add_argument('--drop_last', default=True, type=_str_to_bool)
    parser.add_argument('--shuffle_train', default=True, type=_str_to_bool)

    #评估相关
    parser.add_argument('--roll_step', default=1, type=int,help='rollout step')
    parser.add_argument('--save_pt_dir', default='results/2step_replay_74epoch', type=str,help='save pt directory')
    parser.add_argument('--save_pt', default=False, action='store_true',help='save pt')
    parser.add_argument('--full_finetune', default=False, action='store_true',help='full finetune')
    parser.add_argument('--train_aux', default=False, action='store_true',
                        help='同时训练 LoRA / token_embeds / heads / hypernetwork（默认只FISH backbone）')


    return _finalize_variable_config(parser.parse_args())
