import os
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def keywords_to_text(keywords_docs):
    """
    将每篇文档提取出的关键词转换为文本字符串，关键词之间以空格分隔
    """
    texts = []
    for keywords in keywords_docs:
        # 这里只保留关键词，不包含分数
        text = " ".join([word for word, _ in keywords])
        texts.append(text)
    return texts


def load_hownet_sentiment(folder_path):
    """
    加载知网Hownet情感词典文件夹下的所有文件，并返回一部字典，
    键为词，值为情感分数。
    根据文件名确定词语类型和对应的权重：
      - 程度级别词语: 设置为权重 0 （通常用于调整其他情感词的程度，此处暂不直接计入情感得分）
      - 负面评价词语: 权重 -1
      - 负面情感词语: 权重 -1
      - 正面评价词语: 权重 1
      - 正面情感词语: 权重 1
      - 主张词语: 权重 0.5
    假设每个文件中每行包含一个词（不含显式情感分数）。
    """
    sentiment_dict = {}
    # 文件名与对应的默认权重
    weight_map = {
        "程度级别词语（中文）": 0,
        "负面评价词语（中文）": -1,
        "负面情感词语（中文）": -1,
        "正面评价词语（中文）": 1,
        "正面情感词语（中文）": 1,
        "主张词语（中文)": 0.5
    }
    # 遍历文件夹中所有txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # 根据文件名（去除扩展名）确定词语类型
            category = os.path.splitext(filename)[0]
            weight = weight_map.get(category, 0)  # 若未匹配则设为0
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='ANSI') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        # 如果词已存在，则保留其累计值（这里也可以设计累加或覆盖策略）
                        if word in sentiment_dict:
                            # 例如简单叠加再取平均
                            sentiment_dict[word] = (sentiment_dict[word] + weight) / 2
                        else:
                            sentiment_dict[word] = weight
    return sentiment_dict


def compute_hownet_sentiment_feature(text, sentiment_dict):
    """
    计算文本中知网Hownet情感词的情感得分之和，
    遍历文本中每个词，累加其对应的情感分数（如果存在于情感词典中）。
    """
    score = 0.0
    words = text.split()
    for word in words:
        if word in sentiment_dict:
            score += sentiment_dict[word]
    return score


def build_feature_matrix(texts, hownet_sentiment_file):
    """
    构造特征向量矩阵:
    - 利用 TfidfVectorizer 对关键词文本构造 TF-IDF 特征
    - 使用知网Hownet情感词典计算情感得分作为额外的情感特征
    - 将二者拼接在一起生成最终的特征矩阵
    """
    # 构造 TF-IDF 特征
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_tfidf = vectorizer.fit_transform(texts)

    # 加载 Hownet情感词典
    sentiment_dict = load_hownet_sentiment(hownet_sentiment_file)
    # 计算文本对应的情感得分
    sentiment_features = [compute_hownet_sentiment_feature(text, sentiment_dict) for text in texts]
    sentiment_features = np.array(sentiment_features).reshape(-1, 1)

    # 拼接特征
    X_combined = hstack([X_tfidf, sentiment_features])
    return X_combined, vectorizer