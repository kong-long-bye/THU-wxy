import os
import json
import multiprocessing as mp

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import config
from train import (
    _setup_console_logging,
    preprocess_data,
    preprocess_val_data,
    set_seed,
    split_train_val_by_last_month,
)


def add_ranking_relevance(df):
    # LambdaRank按日期分组，必须先保证同一天的股票连续排列
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    df['rank_desc'] = df.groupby('日期')['label'].rank(method='first', ascending=False)
    df['return_pct'] = df.groupby('日期')['label'].rank(method='average', pct=True)
    # 相关性等级越高，表示该股票在当天越值得排在前面
    df['relevance'] = 0
    df.loc[df['return_pct'] >= 0.70, 'relevance'] = 1
    df.loc[df['return_pct'] >= 0.90, 'relevance'] = 2
    df.loc[df['rank_desc'] <= 5, 'relevance'] = 3
    return df


def main():
    set_seed(config.get('seed', 42))
    _setup_console_logging()

    output_dir = config['lgbm_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}

    # 复用原训练代码的特征工程，避免生成两套不一致的特征
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # instrument仅用于标识股票，不能作为数值特征送入LightGBM
    features = [feature for feature in features if feature != 'instrument']
    train_data['日期'] = pd.to_datetime(train_data['日期'])
    val_data['日期'] = pd.to_datetime(val_data['日期'])
    val_data = val_data[val_data['日期'] >= val_start].copy()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)

    train_data = add_ranking_relevance(train_data)
    val_data = add_ranking_relevance(val_data)

    # group记录每天的股票数量，限定模型只在同一交易日内比较股票
    train_group = train_data.groupby('日期', sort=False).size().to_numpy()
    val_group = val_data.groupby('日期', sort=False).size().to_numpy()

    model = lgb.LGBMRanker(**config['lgbm_params'])
    model.fit(
        train_data[features],
        train_data['relevance'].astype('int32'),
        group=train_group,
        eval_set=[(val_data[features], val_data['relevance'].astype('int32'))],
        eval_group=[val_group],
        eval_metric='ndcg',
        eval_at=(5,),
        callbacks=[
            lgb.early_stopping(config['lgbm_early_stopping_rounds']),
            lgb.log_evaluation(10),
        ],
    )

    # 连同特征顺序一起保存，预测时必须使用完全相同的列
    joblib.dump(
        {'model': model, 'features': features},
        os.path.join(output_dir, 'model.pkl'),
    )

    # 使用真实未来收益计算验证区间内每日Top5的平均表现
    val_data['score'] = model.predict(val_data[features])
    val_top5 = (
        val_data.sort_values(['日期', 'score'], ascending=[True, False])
        .groupby('日期', sort=False)
        .head(5)
    )
    val_top5[['日期', '股票代码', 'label', 'score']].to_csv(
        os.path.join(output_dir, 'validation_top5.csv'), index=False
    )
    best_score = val_top5.groupby('日期')['label'].mean().mean()

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f'Best iteration: {model.best_iteration_}\n')
        f.write(f'Best pred_top5_return: {best_score:.6f}\n')

    print(f'\n训练完成！最佳迭代: {model.best_iteration_}, pred_top5_return: {best_score:.6f}')
    return best_score


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f'\n########## 训练完成！最佳 final score: {best_score:.4f} ##########')
