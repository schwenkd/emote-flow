import os
import cv2
import numpy as np
import tritonhttpclient
from scipy.special import softmax
from icrawler.builtin import GoogleImageCrawler


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    def __init__(self, triton_url='triton:8002'):
        self.input_name = 'Input'
        self.output_name = 'Output'
        self.model_name = 'pet-bokeh'
        self.model_version = '1'
        self.label = 0
        self.input_size = 160
        self.triton_client = tritonhttpclient.InferenceServerClient(url=triton_url, verbose=False)

    def predict(self, img):
        input0 = tritonhttpclient.InferInput(self.input_name, (1, 160, 160, 3), 'FP32')
        input0.set_data_from_numpy(img, binary_data=False)
        output = tritonhttpclient.InferRequestedOutput(self.output_name, binary_data=False)
        response = self.triton_client.infer(self.model_name,
                                            model_version=self.model_version,
                                            inputs=[input0],
                                            outputs=[output])
        logits = response.as_numpy(self.output_name)
        return logits

    def get_mask(self, image):
        #We compute the size of the image to input and resize the image
        target_size = (self.input_size, self.input_size)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        #We run the model and get the segmentation results
        batch_seg_map = self.predict(resized_image[np.newaxis, :, :, :].astype(np.float32))
        #We return the channel corresponding to the pet segmentation
        return batch_seg_map[0, :, :, self.label]

    def transform(self, image, mask, query):
        #We resize the mask to match the original image
        mask = cv2.resize(mask, image.shape[:2][::-1])[:, :, np.newaxis]
        #We get the image size
        x0, y0, c0 = image.shape
        #If the query is "bokeh" we blur the background, if we get a different query we crawl google for an image
        if query != 'bokeh':
            #We create a folder to download the image. We need try-except if the folder already exists
            try:
                os.mkdir(query)
            except:
                pass
            #We run the crawler and download 1 image: https://pypi.org/project/icrawler/
            google_crawler = GoogleImageCrawler(storage={'root_dir': f'/tmp/{query}'})
            google_crawler.crawl(keyword=query, max_num=1)
            #We load the saved image
            background = cv2.imread(f'/tmp/{query}/000001.jpg')
            #We get the background size
            x, y, c = background.shape
            #We resize the background in order to match the original image but keeping aspect ratio
            new_x = x * y0 / y
            new_y = y * x0 / x
            if new_x > x0:
                new_y = y0
            else:
                new_x = x0
            background = cv2.resize(background, (int(new_y), int(new_x)))[:x0, :y0]
        else:
            #The background should be the same image but blurred. We blur the image
            background = cv2.blur(image.copy(), (x0 // 10, y0 // 10))
        #We blend both images using the segmentation mask
        new_img = image * mask + background * (1 - mask)
        #We return the transformed image
        return new_img
