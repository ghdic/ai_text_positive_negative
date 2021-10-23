import os
from flask import Flask, request
from flask_restx import Api, Resource

app = Flask(__name__)
api = Api(app)

@api.route('/shopping')
class Shopping(Resource):
    def get(self):
        return 'shopping' + request.args.get('name')

    def post(self):
        self.get()




@api.route('/movie', methods=['GET', 'POST'])
class Movie(Resource):
    def get(self):
        return 'movie'

    def post(self):
        self.get()



if __name__ == "__main__":
  port = int(os.environ.get("PORT", "5000"))
  app.run(host="0.0.0.0", port=port)
