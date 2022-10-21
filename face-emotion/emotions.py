import tritonhttpclient
import numpy as np
import cv2


class Sentiment():

    def __init__(self, face_model_path, triton_url='triton:8002'):
        """
        We instantiate the Sentiment class with the pretrained model paths
        Args:
            - face_model_path (str): path to the haar cascade opencv model
        """
        self.input_name = 'conv2d_8_input'
        self.output_name = 'dense_5'
        self.model_name = 'face-emotion'
        self.model_version = '1'
        self.triton_client = tritonhttpclient.InferenceServerClient(url=triton_url, verbose=False)
        self.emotion_dict = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"
        }
        self.face_model = cv2.CascadeClassifier(face_model_path)

    def predict(self, img):
        input0 = tritonhttpclient.InferInput(self.input_name, (1, 48, 48, 1), 'FP32')
        input0.set_data_from_numpy(img, binary_data=False)
        output = tritonhttpclient.InferRequestedOutput(self.output_name, binary_data=False)
        response = self.triton_client.infer(self.model_name,
                                            model_version=self.model_version,
                                            inputs=[input0],
                                            outputs=[output])
        logits = response.as_numpy(self.output_name)
        return logits

    def transform(self, frame):
        """
        The predict method should process the image and return the transformed image with the faces and emotions detected.
        Args:
            - frame (np.array): The image as a numpy array with shape (W,H,3)
        Returns:
            - (np.array): Transformed image 
        """
        #The model was trained on grayscale images, therefore we must convert the color image (W,H,3) into grayscale (W,H,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #The face model returns a list of bounding boxes (x,y,w,h) for every detected face
        faces = self.face_model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            #We draw a blue rectangle corresponding to the bounding box
            #https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

            #We use slicing to extract the portion of the image inside the bounding box
            roi_gray = gray[y:y + h, x:x + w]

            #We resize the image to a size of (48,48) which is the input of the model
            resized_roi = cv2.resize(roi_gray, (48, 48))

            #We need to add a batch dimension and a channel dimension
            #The input to the model should have shape (1,48,48,1)
            cropped_img = np.expand_dims(np.expand_dims(resized_roi, -1), 0)

            #We run the emotion detection model and get the softmax output
            prediction = self.predict(cropped_img.astype(np.float32))

            #We get the name of the emotion from the model's output
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]

            #We add the emotion text to the bounding box
            #https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
            cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        return frame