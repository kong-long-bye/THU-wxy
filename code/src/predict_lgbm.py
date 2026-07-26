import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd

from config import config
from predict import preprocess_predict_data


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path = os.path.join(config['lgbm_output_dir'], 'model.pkl')
    output_path = os.path.join('./output/', 'result.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    # 复用原预测代码的特征工程，保证训练和预测口径一致
    processed, _ = preprocess_predict_data(raw_df, stockid2idx)

    artifact = joblib.load(model_path)
    model = artifact['model']

    # 使用训练时保存的特征及顺序，避免预测列错位
    features = artifact['features']

    # 只对最新交易日的完整股票池进行横截面排序
    latest_df = processed[processed['日期'] == latest_date].copy()
    latest_df[features] = latest_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest_df['score'] = model.predict(latest_df[features])

    # 最终选择模型得分最高的5只股票并等权配置
    top5 = latest_df.sort_values('score', ascending=False).head(5)
    if len(top5) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(top5)} 只')

    output_df = pd.DataFrame({
        'stock_id': top5['股票代码'].tolist(),
        'weight': [0.2] * len(top5),
    })
    output_df.to_csv(output_path, index=False)

    print(f'预测日期: {latest_date.date()}')
    print(f'参与排序股票数: {len(latest_df)}')
    print(f'结果已写入: {output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
