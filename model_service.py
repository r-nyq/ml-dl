from flask import Flask, jsonify

from joblib import load

app = Flask(__name__)

MODEL = load("price_model.joblib")

@app.route("/predict_price/<int:area>", methods=["GET"])
def predict_house_price(area):
    prediction = MODEL.predict([[area]])
    return jsonify({"price": prediction[0]})

if __name__ == "__main__":
    with app.app_context():
        app.run(host="0.0.0.0", debug=True)
