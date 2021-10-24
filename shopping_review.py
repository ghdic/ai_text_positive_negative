import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from eunjeon import Mecab

class ShoppingReview():
    def __init__(self):
        self.loaded_model = load_model('shopping_best_model.h5')
        with open('shopping_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def sentiment_predict(self, new_sentence):
        mecab = Mecab()
        stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
        max_len = 80

        new_sentence = mecab.morphs(new_sentence) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = self.tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(self.loaded_model.predict(pad_new)) # 예측
        return score
