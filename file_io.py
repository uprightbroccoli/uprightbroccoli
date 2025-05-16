def read_text(file_path):
    """
    读取文本文件内容，每一行构成一个列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    return None

def write_text_file(file_path, content):
    """
    将内容列表写入文本文件，每个元素一行
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in content:
                file.write(line + "\n")
        print(f"数据已成功保存为 '{file_path}'")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def series_to_txt(series, file_path):
    """
    将 pandas.Series 中的内容写入文本文件
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            for content in series:
                file.write(content + "\n")
        print(f"数据已成功保存为 '{file_path}'")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

# 扩展 pandas Series 方法
def enable_series_to_txt():
    import pandas as pd
    pd.Series.to_txt = series_to_txt

if __name__ == "__main__":
    enable_series_to_txt()
    # 测试写入功能
    sample_series = ["line1", "line2", "line3"]
    sample_file = "sample_output.txt"
    sample_series.to_txt(sample_file)