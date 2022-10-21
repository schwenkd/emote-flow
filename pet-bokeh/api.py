from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from deeplab import DeepLabModel
import numpy as np
import cv2
import io

#We instantiate a deeplab model with the location of the pretrained models
#https://github.com/tensorflow/models/tree/master/research/deeplab
triton_url = 'triton:8000'
model = DeepLabModel(triton_url)

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Pet Bokeh')


#The face-bokeh endpoint receives post requests with the image and returns the transformed image
@app.post("/pet-bokeh/{query}", tags=["Pet Bokeh"])
async def bokeh(file: UploadFile = File(...), query: str = ''):
    #We read the file and decode it
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #We run the model to get the segmentation masks
    mask = model.get_mask(img)
    #We add the background
    return_img = model.transform(img, mask, query)
    #We encode the image before returning it
    _, png_img = cv2.imencode('.PNG', return_img)
    return StreamingResponse(io.BytesIO(png_img.tobytes()), media_type="image/png")


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}
