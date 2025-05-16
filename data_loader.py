import pandas as pd

def load_csv(file_path):
    """
    读取 CSV 文件，并返回 DataFrame
    """
    df = pd.read_csv(file_path)
    print("初始数据基本信息：")
    df.info()
    unique_question_count = df["问题标题"].nunique()
    print(f"问题标题的数量：{unique_question_count}")
    return df
