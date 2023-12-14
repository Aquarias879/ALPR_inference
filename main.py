from flask import Flask, request, jsonify
from license_detect import ObjectDetector
import cv2
import os
import numpy as np
import time

# define model path and classnames
onnx_model1 = "./static/models/yolov8_detect.onnx"  # plate detect
onnx_model2 = "./static/models/best.onnx"  # text
class_names = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
detected_classes = []

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, "./static/upload")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the image file from the request
        upload = request.files['image']

        # Checka valid filename
        if upload.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the upload path
        filename = os.path.join(UPLOAD_PATH, upload.filename)
        upload.save(filename)
        image = cv2.imread(filename)
        
        #license plate detection
        detector1 = ObjectDetector(onnx_model1, class_names,conf_thres=0.2, iou_thres=0.3)
        image = cv2.imread(filename)
        boxes1, scores1, class_ids1 = detector1(image)

        # Loop to crop image pos
        cropped_images = []
        for i, box in enumerate(boxes1):
            x, y, w, h = box
            cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
            cropped_images.append(cropped_image)
            
        detected_classes = []
        for cropped_image in cropped_images:
        #text extraction
            extract_text = ObjectDetector(onnx_model2, class_names, conf_thres=0.3, iou_thres=0.4)
            boxes_text, scores_text, class_ids_text = extract_text(cropped_image)

            #sorting predic character
            sorted_preds = sorted(zip(boxes_text, class_ids_text), key=lambda x: x[0][0])

            #map class ids and class names
            detected_classes.extend([class_names[class_id] for _, class_id in sorted_preds])
            license_number = ''.join(detected_classes) #take data out of list
            detected_classes.clear()
           
            timestamp = str(int(time.time()))  
            filename = f"{license_number}_{timestamp}.png"
            
            #the saved image
            output_path = os.path.join("./static/predict/", filename)
            cv2.imwrite(output_path, cropped_image)

            return jsonify({'success': True, 'result': license_number}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)