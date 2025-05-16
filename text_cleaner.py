import re
import pandas as pd

def clean_text(text):
    if pd.isnull(text):
        return ""

    text = re.sub(r'<.*?>', '', text)  # HTML标签
    text = re.sub(r'&[a-z]+;', '', text)  # HTML转义字符
    text = re.sub(r'http\S+|www\S+', '', text)  # URL链接
    text = re.sub(r'pic\.\S+|img\.\S+', '', text)  # 图片链接
    text = re.sub(r'↓', '', text)
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'\s+', ' ', text)  # 多余空白符
    return text.strip()

