import base64
from io import BytesIO
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse

parser = reqparse.RequestParser()
parser.add_argument('img1', type=str, help='Base64 encoded image 1')
parser.add_argument('img2', type=str, help='Base64 encoded image 2')


def decode_image(base64_string):
    """Decode base64 string to numpy array."""
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return img


def image_compare(img1_base64: str, img2_base64: str) -> bool:
    img1 = decode_image(img1_base64)
    img2 = decode_image(img2_base64)

    # Detect faces in both images
    face_locations_img1 = face_recognition.face_locations(img1)
    face_locations_img2 = face_recognition.face_locations(img2)

    if not face_locations_img1 or not face_locations_img2:
        # No faces detected in one or both images
        return False

    # Compute face encodings
    register_img_encoding = face_recognition.face_encodings(img1, known_face_locations=face_locations_img1)[0]
    scan_img_encoding = face_recognition.face_encodings(img2, known_face_locations=face_locations_img2)[0]

    # Compare face encodings with a tolerance
    result = face_recognition.compare_faces([register_img_encoding], scan_img_encoding, tolerance=0.6)
    is_matched = result[0]

    return is_matched


app = Flask(__name__)
api = Api(app)


class FaceRecognitionApi(Resource):
    def post(self):
        args = parser.parse_args()
        img1_base64, img2_base64 = args['img1'], args['img2']
        try:
            is_matched = image_compare(img1_base64, img2_base64)
            return jsonify({'isMatched': int(is_matched)})  # Convert boolean to integer
        except Exception as e:
            return jsonify({'error': str(e)})


api.add_resource(FaceRecognitionApi, '/predict', endpoint='model')

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
