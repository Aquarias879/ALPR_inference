import os
import requests
import json
import cv2

addr = 'http://192.168.2.140:5000'
upload_url = addr + '/upload'
image_folder = './image/'

# Get a list of all JPEG files in the image folder
jpeg_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Iterate through each JPEG file and send it to the Flask application
for filename in jpeg_files:
    # Read an image using OpenCV
    img = cv2.imread(os.path.join(image_folder, filename))

    # Encode the grayscale image as JPEG
    _, img_encoded = cv2.imencode('.jpg', img)

    # Convert the encoded image to bytes
    img_bytes = img_encoded.tobytes()

    # Send the HTTP request with the image
    response = requests.post(upload_url, files={'image': (filename, img_bytes, 'image/jpeg')})

    # Parse the JSON response
    result = json.loads(response.text)

    # Print the result
    print(f"Result for {filename}: {result}")