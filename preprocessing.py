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


# 定义需要抽取关键词的词性，比如名词(n)、人名(nr)、地名(ns)、机构名(nt)、动名词(vn)
desired_pos = {"n", "nr", "ns", "nt", "vn"}


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
        if word not in stopwords and pos in desired_pos:
            tf[word] += 1
    return tf


def compute_tf_idf(tf, doc_count, word_doc_freq):
    """
    根据单个文档的词频(tf)以及整体文档统计(word_doc_freq)计算 TF-IDF 值。
    doc_count 为总文档数。
    """
    tf_idf = {}
    for word, freq in tf.items():
        idf = math.log((doc_count + 1) / (word_doc_freq[word] + 1)) + 1  # 采用平滑处理
        tf_idf[word] = freq * idf
    return tf_idf


def extract_keywords(input_file, stopwords_file, top_k=10):
    """
    从输入文本文件中每行读取一篇文档，并加载停用词文件，
    最后提取每篇文档中 TF-IDF 最高的 top_k 个关键词。
    """
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
            if word not in stopwords and pos in desired_pos:
                if word not in seen:
                    word_doc_freq[word] += 1
                    seen.add(word)

    doc_count = len(docs_words)

    # 对每个文档计算 TF-IDF，并选出得分最高的关键词
    keywords_docs = []
    for words in docs_words:
        tf = compute_tf(words, stopwords)
        tf_idf = compute_tf_idf(tf, doc_count, word_doc_freq)
        sorted_keywords = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        keywords_docs.append(sorted_keywords)

    return keywords_docs


if __name__ == "__main__":
    input_file = "C:\\Users\\BeLik\\Desktop\\zhihu_analysis\\output_file\\words.txt"  # 原始文本文件，每行一篇文档
    stopwords_file = "C:\\Users\\BeLik\\Desktop\\zhihu_analysis\\tool_file\\stopwords.txt"  # 停用词文件，每行一个停用词
    keywords_docs = extract_keywords(input_file, stopwords_file, top_k=10)

    for i, keywords in enumerate(keywords_docs, 1):
        print(f"第{i}个文档的关键词：")
        for word, score in keywords:
            print(f"{word}: {score:.4f}")
        print()