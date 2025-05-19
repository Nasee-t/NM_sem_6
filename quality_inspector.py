import cv2 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('model/food_model_three.h5')
classes = ['A - Premium', 'B - Acceptable', 'C - Reject']

def preprocess(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Load image:
image_path = 'tomato1.jpg'
# image_path = 'tomato2.jpg'
# image_path = 'tomato3.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or cannot be loaded.")
else:
    input_img = preprocess(image)
    preds = model.predict(input_img)
    label = classes[np.argmax(preds)]

    # Label text on image
    cv2.putText(image, f"Quality: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Food Quality Inspection", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
