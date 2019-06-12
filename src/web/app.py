from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/interpolate', methods=["POST"])
def update_interpolation():
    print(request.json["values"])
    return '', 200


if __name__ == '__main__':
    app.run(debug=True)
