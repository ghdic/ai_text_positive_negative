import os
from flask import Flask, request
from flask_restx import Api, Resource
from movie_review import Movie_Review

app = Flask(__name__)
api = Api(app)

review_movie = Movie_Review()

@api.route('/shopping')
class Shopping(Resource):
    def get(self):
        return 'shopping' + request.args.get('name')

    def post(self):
        self.get()




@api.route('/movie', methods=['GET', 'POST'])
class Movie(Resource):
    def get(self):
        print(request.args.get('q'))
        print(review_movie.sentiment_predict(request.args.get('q')))
        predict = review_movie.sentiment_predict(request.args.get('q'))
        return {'predict': predict}

    def post(self):
        self.get()



if __name__ == "__main__":
  port = int(os.environ.get("PORT", "5000"))
  app.run(host="0.0.0.0", port=port)
