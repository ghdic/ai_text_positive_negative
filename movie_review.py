import re
from konlpy.tag import Okt
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)

okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
max_len = 30


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
