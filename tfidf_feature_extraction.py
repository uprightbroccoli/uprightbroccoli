import math
from collections import defaultdict


def load_stopwords(filepath):
    """
    从停用词文件加载停用词，每行一个停用词，并返回一个集合
    """
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:  # 忽略空行
                stopwords.add(word)
    return stopwords




def parse_line(line):
    """
    解析一行标注好的文本，返回一个词的列表，格式为 (word, pos)。
    假设每个词的格式为：词/词性。
    """
    tokens = line.strip().split()
    words = []
    for token in tokens:
        if "/" in token:
            word, pos = token.split("/", 1)
            words.append((word, pos))
    return words


def compute_tf(words, stopwords):
    """
    计算单个文档的词频，只统计满足条件（不在停用词中且词性匹配 desired_pos）的词
    """
    tf = defaultdict(int)
    for word, pos in words:
        if word not in stopwords:
            tf[word] += 1
    return tf


def compute_tf_idf(tf, doc_count, word_doc_freq):
    """
    根据单个文档的词频(tf)以及整体文档统计(word_doc_freq)计算 TF-IDF 值。
    doc_count 为总文档数。
    使用平滑处理防止除0错误。
    """
    tf_idf = {}
    for word, freq in tf.items():
        idf = math.log((doc_count + 1) / (word_doc_freq[word] + 1)) + 1
        tf_idf[word] = freq * idf
    return tf_idf


def extract_tfidf_features(input_file, stopwords_file):
    """
    从输入文本文件中每行读取一篇文档，并加载停用词文件，
    计算每篇文档中每个词的 TF-IDF 值，最后形成 TF-IDF 特征向量。
    特征向量的维度等于整体语料库中所有非停用词且词性符合条件的词汇数。
    """
    # 加载停用词
    stopwords = load_stopwords(stopwords_file)

    docs_words = []  # 存储每个文档的词列表
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            words = parse_line(line)
            docs_words.append(words)

    # 统计每个词出现在多少个文档中
    word_doc_freq = defaultdict(int)
    for words in docs_words:
        seen = set()
        for word, pos in words:
            if word not in stopwords:
                if word not in seen:
                    word_doc_freq[word] += 1
                    seen.add(word)

    doc_count = len(docs_words)

    # 构建词汇表（按字母顺序排序）
    vocabulary = sorted(word_doc_freq.keys())
    word2index = {word: idx for idx, word in enumerate(vocabulary)}

    features = []  # 存储每个文档的 TF-IDF 向量
    for words in docs_words:
        tf = compute_tf(words, stopwords)
        tf_idf = compute_tf_idf(tf, doc_count, word_doc_freq)
        # 初始化向量，维度等于 vocabulary 的长度
        doc_vector = [0.0] * len(vocabulary)
        for word, value in tf_idf.items():
            if word in word2index:
                index = word2index[word]
                doc_vector[index] = value
        features.append(doc_vector)

    return features, vocabulary


if __name__ == "__main__":

    input_file = "output_file\\words.txt"  # 每行一篇文档的文件
    stopwords_file = "tool_file\\stopwords.txt"  # 停用词文件，每行一个停用词
    features, vocabulary = extract_tfidf_features(input_file, stopwords_file)

    print("Vocabulary size:", len(vocabulary))
    print("TF-IDF feature vector for the first document:")
    print(features[0])