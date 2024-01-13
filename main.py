import os
import json
import requests
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class Load_Data_Request(BaseModel):
    data: str



@app.get('/'):
def root():
    return {"message": "/docs for more info"}

@app.get('/data')
def download_file(load_data_request: Load_Data_Request):
    url = load_data_request.data
    filename = url.split("/")[-1]
    response = requests.get(url)
        # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

class TrainModelRequest(BaseModel):
    model: str
    data_filepath: str
    
@app.get('/train')
def train_model():
    