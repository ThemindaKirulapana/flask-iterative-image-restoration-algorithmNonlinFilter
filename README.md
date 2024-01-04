
# Flask App for Image Restoration

When you upload an image, you can view the restored verison of it using this app.

apply_nonlinear_filter: This function takes an input image and applies a non-linear filter, specifically a median filter using OpenCV's cv2.medianBlur function.

random_sampling: This function performs random sampling on the filtered image. It calculates the number of pixels to sample based on a fraction of the total pixels, then randomly selects pixel coordinates.

## Installation

First clone the project and open the folder using an IDE like VS Code, Pycharm.
Then open the terminal and create a virtual environment.

```bash
  python -m venv env
```
    
Then activate virtual environment

```bash
  env/Scripts/activate
``` 
Next install the required packages.

```bash
  pip install -r requirements.txt
``` 
Create a 'static' folder in the root directory.
Run the app

```bash
  python app.py
``` 
