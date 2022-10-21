from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from emotions import Sentiment
import numpy as np
import cv2
import io

#We instantiate a face-emotion detector with the location of the pretrained models
triton_url = 'triton:8000'
face_model_path = './haarcascade_frontalface_default.xml'
model = Sentiment(face_model_path, triton_url)

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Face Emotion')


#The face-sentiment endpoint receives post requests with the image and returns the transformed image
@app.post("/face-sentiment", tags=["Sentiment Analysis"])
async def sentiment(file: UploadFile = File(...)):
    #We read the file and decode it
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #We run the model to get the transformed image
    return_img = model.transform(img)

    #We encode the image before returning it
    _, png_img = cv2.imencode('.PNG', return_img)
    return StreamingResponse(io.BytesIO(png_img.tobytes()), media_type="image/png")


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}
