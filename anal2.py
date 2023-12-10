from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import re
from nltk.corpus import stopwords

# Создаем экземпляр CountVectorizer с указанием языка
max_feature = 400
vectorizer = CountVectorizer(token_pattern=r'\b[а-яА-ЯёЁ]+\b', max_features=max_feature)

# Читаем положительные и отрицательные тексты из файлов
with open("pos.txt", "r", encoding='utf-8') as f:
    positive_texts = f.readlines()
with open("neg.txt", "r", encoding='utf-8') as f:
    negative_texts = f.readlines()

stop_words_set = set(stopwords.words('russian'))
stop_words_set.add('это')
stop_words_set.add('оно') 
stop_words_set.add('эта')
#print(stop_words_set)

# Функция для проверки однокоренных слов
def is_related(word1, word2):
    return word1[:-2] == word2[:-2]

# Функция для фильтрации информативных слов
def filter_top_words(top_words):
    filtered_words = []
    for weight, word in top_words:
        if (word not in stop_words_set and not re.match(r'^\d', word) and
                not any(is_related(word, filtered_word) for _, filtered_word in filtered_words)):
            filtered_words.append((weight, word))
    return filtered_words

# Создаем тренировочную выборку
X_train = positive_texts + negative_texts
y_train = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train)

# Формируем матрицу признаков из тренировочной выборки
X_train_vectorized = vectorizer.fit_transform(X_train)

# Создаем и обучаем нейронную сеть
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, verbose = True, learning_rate_init=0.0001, batch_size=25, alpha=0.1)
clf.fit(X_train_vectorized, y_train)

# Читаем тестовые тексты из файла
with open("test_texts.txt", "r", encoding='utf-8') as f:
    test_texts = f.readlines()

# Формируем матрицу признаков из тестовой выборки
X_test_vectorized = vectorizer.transform(test_texts)

# Предсказываем тональность текстов
y_pred = clf.predict(X_test_vectorized)
y_pred_proba = clf.predict_proba(X_test_vectorized)
total_accuracy = np.mean([1 if true == pred else 0 for true, pred in zip(y_test, y_pred)])

a = 0
b = 0
# Выводим результаты
for text, pred, prob in zip(test_texts, y_pred, y_pred_proba):
    if pred == 'positive':
        #print(f"Текст: {text.strip()}")
        #print(f"Тональность:{'%.2f' % (prob[1]*100)}% положительный")
        a = float('%.2f' % (prob[1]*100)) + a
    else:
        #print(f"Текст: {text.strip()}")
        #print(f"Тональность:{'%.2f' % (prob[0]*100)}% негативный")
        b = float('%.2f' % (prob[0]*100)) + b
print(f"Общая точность = {(a+b)/20}%")
#print(f"Общее количество правильно распознанных отзывов: {int(total_accuracy * len(y_test))} из {len(y_test)}")


# Находим наиболее информативные слова и их веса
feature_names = vectorizer.get_feature_names()

# Фильтруем информативные слова, исключая числовые значения, стоп-слова и однокоренные слова
top_words = [(weight, word) for weight, word in zip(clf.coefs_[-1], feature_names)]
top_words = filter_top_words(top_words)
top_words = sorted(top_words, reverse=True)

#print("\nНаиболее информативные слова:")
#for weight, word in top_words:
#    print(f"{word}: 𝜔={weight}")

# Формируем словарь с количеством вхождений каждого слова в тренировочной выборке
word_counts = {}
for i, word in enumerate(feature_names):
    if word not in stop_words_set and not re.match(r'^\d', word):
        filtered = False
        for _, filtered_word in top_words:
            if is_related(word, filtered_word):
                filtered = True
                break
        if not filtered:
            word_counts[word] = word_counts.get(word, 0) + X_train_vectorized.getcol(i).sum()
                                                                                        
# Находим наиболее частоповторяющиеся слова и их количество среди информативных слов
#top_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:30]
#print("\nНаиболее часто повторяющиеся слова среди информативных:")
#for word, count in top_counts:
#    print(f"{word}: {count}")