from flask import Flask, render_template, request, send_file
from src.web.interpolation_handler import InterpolationHandler
import io

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/interpolate', methods=["POST"])
def update_interpolation():
    image_binary = interpolation_handler.attempt_interpolation(request.json["values"])
    if image_binary is not None:
        return send_file(
           io.BytesIO(image_binary),
           mimetype='image/jpeg',
           as_attachment=True,
           attachment_filename='interpolated_image.jpeg')
    return ""


if __name__ == '__main__':
    interpolation_handler = InterpolationHandler()
    app.run(debug=True)
