import pandas as pd
import re
from hash_utils import calculate_hash
from text_cleaner import clean_text


def filter_and_group(df):
    # 应用文本清洗到“回答内容”
    df.loc[:,"回答内容"] = df["回答内容"].apply(clean_text)

    # 添加哈希列，用于后续数据去重
    df.loc[:,'content_hash'] = df['回答内容'].apply(calculate_hash)

    # 检查并转换赞同数为数值类型
    df.loc[:,"赞同数"] = pd.to_numeric(df["赞同数"], errors='coerce').fillna(0)

    # 按赞同数排序并去重
    df = df.sort_values('赞同数', ascending=False).drop_duplicates(subset='content_hash').drop(
        columns=['content_hash']).reset_index(drop=True)

    print("数据处理完成！")

    # 剔除回答内容少于15字的记录
    df = df[df['回答内容'].str.len() >= 15]
    df = df.reset_index(drop=True)
    print(f"剔除过短评论后剩余 {len(df)} 条记录")

    # 定义关键词列表（可根据实际需求调整）
    keywords = [
        "京东外卖", "美团外卖", "京东美团商战", "外卖补贴", "骑手待遇",
        "商家佣金", "配送效率", "品质外卖", "外卖价格战",
        "外卖市场份额", "刘强东", "王兴", "即时零售", "外卖用户体验"
    ]
    keyword_pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    # 筛选包含关键词的记录
    df = df[df['回答内容'].str.contains(keyword_pattern)]
    df = df.reset_index(drop=True)
    print(f"筛选后剩余 {len(df)} 条记录")

    # 按照“问题标题”对回答内容合并
    df_grouped = df.groupby("问题标题")["回答内容"].apply(lambda x: " ".join(x)).reset_index()

    # 对每个问题统计回答数
    question_counts = df["问题标题"].value_counts().reset_index()
    question_counts.columns = ["问题标题", "回答数"]
    question_counts_sorted = question_counts.sort_values(by="回答数", ascending=False)

    return df, df_grouped, question_counts_sorted


if __name__ == "__main__":
    # 示例：加载预处理后的数据，然后过滤和分组
    from data_loader import load_csv
    from preprocess import preprocess_data

    file_path = "feet_file\\raw_data.csv"
    df = load_csv(file_path)
    df_cleaned = preprocess_data(df)
    df_processed, df_grouped, question_counts = filter_and_group(df_cleaned)

    print("问题回答统计预览：")
    print(question_counts.head())