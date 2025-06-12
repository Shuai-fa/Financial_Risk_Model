# file: 1_create_sample_data.py
import pandas as pd
import numpy as np
import os

# --- TODO: 请修改为您真实 train.csv 文件的路径 ---
PATH_TO_YOUR_REAL_TRAIN_CSV = '/Users/imac/Desktop/bank/train.csv' 
# ---------------------------------------------

OUTPUT_SAMPLE_PATH = 'data/sample_data.csv'
N_ROWS = 50 # 创建一个50行的样本

def run():
    """
    Loads real data, creates an anonymized, English-header sample,
    and saves it to the data/ directory.
    """
    print("--- Creating Anonymized English Sample Data ---")

    # 检查data文件夹是否存在
    if not os.path.exists('data'):
        os.makedirs('data')

    try:
        # 1. 加载真实数据
        df = pd.read_csv(PATH_TO_YOUR_REAL_TRAIN_CSV, header=None, low_memory=False)
        df.dropna(how='all', inplace=True)

        # 2. 定义中英文标题映射
        header_map = {
            '是否违约': 'default_status', '货币资金': 'cash', '应收账款': 'accounts_receivable', 
            '存货': 'inventory', '流动资产合计': 'total_current_assets', '固定资产': 'fixed_assets', 
            '非流动资产合计': 'total_non_current_assets', '资产总计': 'total_assets', '短期借款': 'short_term_debt',
            '应付账款': 'accounts_payable', '非流动负债合计': 'total_non_current_liabilities', 
            '所有者权益(或股东权益)合计': 'total_equity', '报告期': 'report_period', '年份': 'year', '月份': 'month',
            '营业收入': 'revenue', '营业成本': 'cost_of_sales', '财务费用': 'finance_expense', 
            '资产减值损失': 'asset_impairment_loss', '营业利润': 'operating_profit', '利润总额': 'total_profit',
            '净利润': 'net_profit', '违约年份': 'default_year', '违约月份': 'default_month', 
            '资产减值占比': 'impairment_ratio', '净利率': 'net_profit_margin', '代码': 'company_id'
        }

        # 3. 校正表头并重命名为英文
        df.columns = df.iloc[0].str.strip()
        df = df.iloc[1:].reset_index(drop=True)
        df.rename(columns=header_map, inplace=True)

        # 4. 取样并匿名化
        sample_df = df.head(N_ROWS).copy()
        sample_df['company_id'] = [f'COMPANY_{i+1:04d}' for i in range(N_ROWS)]

        # 5. 保存样本文件
        sample_df.to_csv(OUTPUT_SAMPLE_PATH, index=False, encoding='utf-8-sig')

        print(f"Success! Anonymized sample file saved to: {OUTPUT_SAMPLE_PATH}")
        print("This file is safe to upload to GitHub.")

    except FileNotFoundError:
        print(f"ERROR: Real training file not found. Please check the path: '{PATH_TO_YOUR_REAL_TRAIN_CSV}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    run()