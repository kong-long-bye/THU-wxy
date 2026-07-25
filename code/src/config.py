# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'loss_temperature': 1.0, # Top5RankingLoss 温度参数
    'pairwise_weight': 0.5, # 配对损失权重

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',


    'lgbm_output_dir': f'./model/lgbm_{feature_num}',
    'lgbm_early_stopping_rounds': 50,
    'lgbm_params': {
        # 直接优化同一交易日内的股票排序
        'objective': 'lambdarank',
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 30,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': 42,
        'n_jobs': -1,
        'importance_type': 'gain',
        # Top5任务重点关注前8名附近的排序
        'lambdarank_truncation_level': 8,
    },
}