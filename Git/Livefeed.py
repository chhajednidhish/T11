import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle

pickle.dump(neigh, open('imgknn_0toz.p','wb')) #to deploy and avoid rerunning
model = pickle.load(open('imgknn_model.p','rb'))


def prediction(pred):
    return(chr(pred+ 65))


def keras_predict(model, image):
        #KNN classifier
        flat_data = []
        url = input('Enter your URL : ')
        img_array = cv2.imread(url,2)   
        ret, bw = cv2.threshold(img_array, 90, 255, cv2.THRESH_BINARY)
        bw_resized = resize(bw, (128,128,1)) #normalizes between 0 to 1
        flat_data.append(bw_resized.flatten())
        flat_data = np.array(flat_data)
        plt.imshow(bw_resized)
        y_out = model3.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        print(f'PREDICTED OUTPUT : {y_out}')
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    l = []
    
    while True:
        
        cam_capture = cv2.VideoCapture(0)
        _, image_frame = cam_capture.read()  
    # Select ROI
        im2 = crop_image(image_frame, 300,300,300,300)
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
        im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)


    
        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)
    

        pred_probab, pred_class = keras_predict(model, im5)
    
        curr = prediction(pred_class)
        
        cv2.putText(image_frame, curr, (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
            
            
    
 
    # Display cropped image
        cv2.rectangle(image_frame, (300, 300), (600, 600), (255, 255, 00), 3)
        cv2.imshow("frame",image_frame)
        
        
    #cv2.imshow("Image4",resized_img)
        cv2.imshow("Image3",image_grayscale_blurred)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()

cam_capture.release()
cv2.destroyAllWindows()