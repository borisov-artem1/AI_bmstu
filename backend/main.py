from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.filling_the_dataset import dataset, taste_tree

if __name__ == "__main__":
    # Загружаем данные в DataFrame
    df = pd.DataFrame(taste_tree)

    # Кодируем категориальные данные (flavor)
    label_encoder = LabelEncoder()
    df['flavor_encoded'] = label_encoder.fit_transform(df['flavor'])

    # Выбираем признаки для классификации (вкусы и типы)
    X = df[['flavor_encoded']]
    y = df['product']

    # Построение дерева решений для вкусов
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X, y)

    # Отображение структуры дерева

    plt.figure(figsize=(20, 10))
    tree.plot_tree(tree_clf, feature_names=['flavor'], class_names=df['product'], filled=True)
    plt.savefig('tree.png', dpi=300)
