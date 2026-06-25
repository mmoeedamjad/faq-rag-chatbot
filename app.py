from flask import Flask, render_template, request, jsonify
from rag_engine import answer_query

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = answer_query(user_message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
