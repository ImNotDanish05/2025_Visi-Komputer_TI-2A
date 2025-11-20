"""
https://huggingface.co/facebook/sam2-hiera-tiny/blob/f245b47be73d8858fb7543a8b9c1c720d9f98779/sam2_hiera_tiny.pt
"""
import base64
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from sam2_utils import segment_and_color


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload guard


def _decode_image_from_request() -> Optional[np.ndarray]:
    """Decode an image sent as multipart 'frame' or JSON base64."""
    if "frame" in request.files:
        data = request.files["frame"].read()
    else:
        payload = request.get_json(silent=True) or {}
        raw_b64 = payload.get("image_base64")
        if not raw_b64:
            return None
        # Accept data URI or plain base64
        b64_str = raw_b64.split(",", 1)[-1]
        data = base64.b64decode(b64_str)

    np_data = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def segment():
    image = _decode_image_from_request()
    if image is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        processed = segment_and_color(image)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Segmentation failed: {exc}"}), 500

    success, buffer = cv2.imencode(".png", processed)
    if not success:
        return jsonify({"error": "Encoding failed"}), 500

    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return jsonify({"image_base64": f"data:image/png;base64,{encoded}"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
