import numpy as np
import pandas as pd
from data_loader import load_csv
from preprocess import preprocess_data
from filter_group import filter_and_group
from file_io import enable_series_to_txt
from preprocessing import extract_keywords
from tfidf_feature_extraction import extract_tfidf_features
from feature_extraction import keywords_to_text, build_feature_matrix
from sentiment_classification import train_and_evaluate, save_model


def main():
    # 1. 加载数据
    file_path = "feet_file\\raw_data.csv"
    df = load_csv(file_path)

    # 2. 数据预处理
    df_cleaned = preprocess_data(df)
    print("预处理后的数据概况：")
    df_cleaned.info()
    # 3. 数据过滤与聚合
    df_processed, df_grouped, question_counts = filter_and_group(df_cleaned)

    # 4. 将合并后的回答内容写入文件
    enable_series_to_txt()
    output_file = "feet_file\\words.txt"
    df_grouped["回答内容"].to_txt(output_file)

    # 5. 输出问题标题数量及回答统计
    question_count = df_grouped["问题标题"].count()
    print(f"最终问题数：{question_count}")
    print("问题回答数统计（前10）：")
    print(question_counts.head(10))

    # 配置文件路径
    input_file = "output_file\\words.txt"  # 原始文本，每行一篇文档（格式为 词/词性 ）
    stopwords_file = "tool_file\\stopwords.txt"  # 停用词文件，每行一个停用词
    hownet_sentiment_file = "tool_file\\知网Hownet情感词典"  # 知网Hownet情感词典文件，每行格式：词 <tab> 情感分数

    # 1. 对文本进行预处理并提取特征值
    features,vocabulary = extract_tfidf_features(input_file, stopwords_file)

    # 2. 将提取关键词转换为文本（以空格分隔）
    texts = keywords_to_text(keywords_docs)

    # 3. 构造基于 TF-IDF 和 Hownet 情感特征的特征向量矩阵
    X, vectorizer = build_feature_matrix(texts, hownet_sentiment_file)
    print("构造的特征向量矩阵形状：", X.shape)

    # 4. 构造情感标签 (示例中 1 代表正面，0 代表负面)
    # 请确保 labels 列表的长度与文本数一致，并根据实际情况准备标签数据
    labels = np.random.randint(0, 2, 246).tolist() # 示例标签

    # 5. 模型训练与评估
    model = train_and_evaluate(X, labels)

    # 6. 保存训练好的模型
    save_model(model, "sentiment_model.pkl")
if __name__ == "__main__":
    main()