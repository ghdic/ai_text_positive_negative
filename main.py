import os
from flask import Flask, request
from flask_restx import Api, Resource
from movie_review import MovieReview
from shopping_review import ShoppingReview

app = Flask(__name__)
api = Api(app)

review_movie = MovieReview()
review_shopping = ShoppingReview()


@api.route('/shopping')
class Shopping(Resource):
    def get(self):
        predict = review_shopping.sentiment_predict(request.args.get('q'))
        print(predict)
        return {'predict': predict}

    def post(self):
        self.get()


@api.route('/movie')
class Movie(Resource):
    def get(self):
        predict = review_movie.sentiment_predict(request.args.get('q'))
        print(predict)
        return {'predict': predict}

    def post(self):
        self.get()


if __name__ == "__main__":
  port = int(os.environ.get("PORT", "5000"))
  app.run(host="0.0.0.0", port=port)
