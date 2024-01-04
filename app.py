from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.signal import convolve2d

app = Flask(__name__)

def iterative_image_restoration(image, num_iterations, alpha, filter_kernel):
    restored_image = np.copy(image)

    for iteration in range(num_iterations):
        filtered_output = convolve2d(restored_image, filter_kernel, mode='same', boundary='symm', fillvalue=0)
        restored_image = restored_image * (1 - alpha) + alpha * (image - filtered_output)

    return restored_image.astype('uint8')

def preprocess_image(file):
    image = Image.open(BytesIO(file.read())).convert("L")
    image_array = np.array(image)
    return image, image_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            original_image, original_image_array = preprocess_image(file)

            # Set parameters for iterative image restoration
            num_iterations = 10
            alpha = 0.1

            # Define a filter kernel (customize based on your requirements)
            filter_kernel = np.array([[0.1, 0.1, 0.1],
                                      [0.1, 0.2, 0.1],
                                      [0.1, 0.1, 0.1]])
            filter_kernel /= np.sum(filter_kernel)

            # Apply iterative image restoration
            restored_image_array = iterative_image_restoration(original_image_array, num_iterations, alpha, filter_kernel)

            # Convert NumPy array back to PIL image for display
            restored_image = Image.fromarray(restored_image_array)

            # Save the restored image
            restored_image_path = "static/restored_image.png"
            restored_image.save(restored_image_path)

            return render_template("index.html", original_image=original_image, restored_image_path=restored_image_path)

    return render_template("index.html", original_image=None, restored_image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
