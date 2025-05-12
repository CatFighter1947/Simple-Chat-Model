import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

# Читаю данные
data_df = pd.read_csv("good.tsv", sep='\t')

# Создаю дополнительный датасет из первых 2ух реплик, откидывая пустые
data_dop_df = data_df[['context_2', 'context_1']].copy()
data_dop_df.dropna(how='any', inplace=True)

# Переименовываю столбцы, чтобы названия были одинаковыми и конкатенация произошла без Nan
data_dop_df.rename(columns={'context_2':'context_0', 'context_1':'reply'}, inplace=True) #

# Выделяю основной датасет
data_df = data_df[['context_0', 'reply']]

# print(data_dop_df.describe())
# print(data_df.describe())

# Произвожу конкатенацию по строкам для увеличения датасета
data_df = pd.concat([data_df, data_dop_df], axis=0 ).reset_index(drop=True)

#print(data_df.sample(5)) # показать 5 случайных строк
# print(data_df.describe())
# print(train.info())

# Векторизация
vectorizer = TfidfVectorizer()
vectorizer.fit(data_df.context_0)
matrix_big = vectorizer.transform(data_df.context_0)
#print(matrix_big.shape)

# Сокращение размерности (метод главных компонент)
svd = TruncatedSVD(n_components=300)
svd.fit(matrix_big)
matrix_small = svd.transform(matrix_big)
#print(matrix_small.shape)
#print(svd.explained_variance_ratio_.sum()) # разобраться потом


# Случайный выбор одного из ближайших соседей
def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)

class NeighborSampler(BaseEstimator):
    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature

    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)

    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
        return self.y_[result]


ns = NeighborSampler()
ns.fit(matrix_small, data_df.reply)

# Создадим pipeline
pipe = make_pipeline(vectorizer, svd, ns)
print(pipe.predict(['сколько тебе лет ?']))