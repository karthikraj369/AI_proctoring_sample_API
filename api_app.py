# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:32:55 2024

@author: LENOVO1
"""

from flask import Flask, Response, jsonify
from mtcnn import MTCNN
import cv2

app = Flask(__name__)

detector = MTCNN()

def detect_faces():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        faces = detector.detect_faces(frame)
        # Extract face coordinates
        face_coords = []
        for face in faces:
            x, y, width, height = face['box']
            face_coords.append({'x': x, 'y': y, 'width': width, 'height': height})

        yield jsonify({'faces': face_coords})

@app.route('/api/faces')
def get_faces():
    return Response(detect_faces(), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
