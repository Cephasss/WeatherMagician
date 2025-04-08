ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 35]
    },
    multires = [1,2],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False

)
OptimizationParams = dict(
    lambda_normal = 0.1,
    lambda_alpha = 0.2,
    lambda_dssim = 0.2,
    opacity_lr = 0.05,
    hard_opt=1,
    soft_opt=1,
    dataloader=False,
    iterations = 30000,
    batch_size=1,
    coarse_iterations = 1,
    densification_interval = 100,
    densify_until_iter = 15_000,
    densify_grad_threshold_coarse = 0.0008,
    densify_grad_threshold_fine_init = 0.0008,
    densify_grad_threshold_after = 0.0006,
    opacity_reset_interval = 461000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.003,
    pruning_interval = 500
)