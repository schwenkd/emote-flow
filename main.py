from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Serverless Lambda FastAPI')


@app.get("/", tags=["Health Check"])
async def root():
    bokeh = requests.get('http://pet-bokeh:8000/')
    emotion = requests.get('http://face-emotion:8000/')
    return {"face-bokeh": bokeh.json()['message'], "face-emotion": emotion.json()['message']}
