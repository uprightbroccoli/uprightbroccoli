from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib


def train_and_evaluate(X, labels):
    """
    划分训练集和测试集，训练逻辑回归模型，并评估模型性能
    """
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上预测并输出分类报告
    y_pred = model.predict(X_test)
    print("测试集上的分类报告：")
    print(classification_report(y_test, y_pred))

    # 5折交叉验证评估模型效果（F1 分数）
    cv_scores = cross_val_score(model, X, labels, cv=5, scoring="f1")
    print("交叉验证 F1 分数：", cv_scores)
    print("平均 F1 分数：", cv_scores.mean())

    return model


def save_model(model, filename):
    """
    将训练好的模型保存到文件中
    """
    joblib.dump(model, filename)
    print(f"模型已保存到 {filename}")