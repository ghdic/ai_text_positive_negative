import pandas as pd
import numpy as np
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)

okt = Okt()
tokenizer = Tokenizer()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
max_len = 30

train_data = pd.read_table('movie_ratings_train.txt')
test_data = pd.read_table('movie_ratings_test.txt')

train_data = train_data.dropna(how='any')  # Null 값이 존재하는 행 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
train_data['document'] = train_data['document'].str.replace('^ +', "",
                                                            regex=True)  # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')

test_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)  # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True)  # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거

X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
    X_train.append(stopwords_removed_sentence)

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
    X_test.append(stopwords_removed_sentence)

tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 1
print('vocab_size', vocab_size)

tokenizer = Tokenizer(vocab_size)  # 빈도수 2 이하인 단어는 제거
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


class Movie_Review():
    def __init__(self):
        self.loaded_model = load_model('movie_best_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def sentiment_predict(self, new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
        new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
        new_sentence = [word for word in new_sentence if word not in stopwords]  # 불용어 제거
        encoded = self.tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
        score = float(self.loaded_model.predict(pad_new))  # 예측
        return score


if __name__ == '__main__':
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename="movie_ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="movie_ratings_test.txt")

    train_data = pd.read_table('movie_ratings_train.txt')
    test_data = pd.read_table('movie_ratings_test.txt')

    train_data = train_data.dropna(how='any')  # Null 값이 존재하는 행 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
    train_data['document'] = train_data['document'].str.replace('^ +', "",
                                                                regex=True)  # white space 데이터를 empty value로 변경
    train_data['document'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how='any')

    test_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)  # 정규 표현식 수행
    test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True)  # 공백은 empty 값으로 변경
    test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any')  # Null 값 제거

    X_train = []
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        X_train.append(stopwords_removed_sentence)

    X_test = []
    for sentence in tqdm(test_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        X_test.append(stopwords_removed_sentence)

    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    vocab_size = total_cnt - rare_cnt + 1
    print('vocab_size', vocab_size)

    tokenizer = Tokenizer(vocab_size)  # 빈도수 2 이하인 단어는 제거
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]

    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)

    X_test = np.delete(X_test, drop_test, axis=0)
    y_test = np.delete(y_test, drop_test, axis=0)

    # 전체 데이터의 길이는 30으로 맞춘다.
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    # mc = ModelCheckpoint('movie_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_weights_only=True)

    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es], batch_size=64, validation_split=0.2)
    model.save('movie_best_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    review = Movie_Review()
    print(review.sentiment_predict('아 ㅋㅋ 이 영화 개꿀잼이자나 ㅋㅋ'))
