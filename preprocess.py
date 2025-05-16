import pandas as pd


def preprocess_data(df):
    # 处理赞同数和评论数
    df["赞同数"] = df["赞同数"].astype(str).str.replace(r"\s|​|,|个|👍|赞|\+", "", regex=True)
    df["评论数"] = df["评论数"].astype(str).str.replace(r"\s|​|,|条|评论", "", regex=True)

    # 将其转换为数字类型，无法转换的设为 NaN
    df["赞同数"] = pd.to_numeric(df["赞同数"], errors='coerce')
    df["评论数"] = pd.to_numeric(df["评论数"], errors='coerce')

    # 标准化“回答时间”字段，提取日期并转换为 datetime 类型
    df["回答时间"] = df["回答时间"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
    df["回答时间"] = pd.to_datetime(df["回答时间"], errors='coerce')

    # 删除完全缺失“回答内容”的记录
    df = df.dropna(subset=["回答内容"])

    # 填充其他缺失值
    df.loc[:,"问题内容"] = df["问题内容"].fillna("")
    df.loc[:,"答主昵称"] = df["答主昵称"].fillna("匿***")
    df.loc[:,'回答时间'] = df['回答时间'].fillna('2025-05-08 16:23:00')
    df.loc[:,'赞同数'] = df['赞同数'].fillna('0')
    df.loc[:,'评论数'] = df['评论数'].fillna('0')

    return df
